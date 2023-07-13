# Remove it after first run to avoid recompilation
#= include("header.jl") =#

# Use the target test header file
include("test/linear_scalar_advection_3d.jl")
#= include("test/compressible_euler_3d.jl") =#

# Kernel configurators 
#################################################################################

# CUDA kernel configurator for 1D array computing
function configurator_1d(kernel::CUDA.HostKernel, array::CuArray{Float32,1})
    config = launch_configuration(kernel.fun)

    threads = min(length(array), config.threads)
    blocks = cld(length(array), threads)

    return (threads=threads, blocks=blocks)
end

# CUDA kernel configurator for 2D array computing
function configurator_2d(kernel::CUDA.HostKernel, array::CuArray{Float32,2})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 2))), 2))
    blocks = map(cld, size(array), threads)

    return (threads=threads, blocks=blocks)
end

# CUDA kernel configurator for 3D array computing
function configurator_3d(kernel::CUDA.HostKernel, array::CuArray{Float32,3})
    config = launch_configuration(kernel.fun)

    threads = Tuple(fill(Int(floor((min(maximum(size(array)), config.threads))^(1 / 3))), 3))
    blocks = map(cld, size(array), threads)

    return (threads=threads, blocks=blocks)
end

# Helper functions
#################################################################################

# Rewrite `get_node_vars()` as a helper function
@inline function get_nodes_vars(u, equations, indices...)

    SVector(ntuple(@inline(v -> u[v, indices...]), Val(nvariables(equations))))
end

# Rewrite `get_surface_node_vars()` as a helper function
@inline function get_surface_node_vars(u, equations, indices...)

    u_ll = SVector(ntuple(@inline(v -> u[1, v, indices...]), Val(nvariables(equations))))
    u_rr = SVector(ntuple(@inline(v -> u[2, v, indices...]), Val(nvariables(equations))))

    return u_ll, u_rr
end

# Rewrite `get_node_coords()` as a helper function
@inline function get_node_coords(x, equations, indices...)

    SVector(ntuple(@inline(idx -> x[idx, indices...]), Val(ndims(equations))))
end

# CUDA kernels 
#################################################################################

# Copy data to GPU (run as Float32)
function copy_to_gpu!(du, u)
    du = CUDA.zeros(size(du))
    u = CuArray{Float32}(u)

    return (du, u)
end

# Copy data to CPU (back to Float64)
function copy_to_cpu!(du, u)
    du = Array{Float64}(du)
    u = Array{Float64}(u)

    return (du, u)
end

# CUDA kernel for calculating fluxes along normal direction 1, 2, and 3
function flux_kernel!(flux_arr1, flux_arr2, flux_arr3, u, equations::AbstractEquations{3}, flux::Function)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^3 && k <= size(u, 5))
        j1 = div(j - 1, size(u, 2)^2) + 1
        j2 = div(rem(j - 1, size(u, 2)^2), size(u, 2)) + 1
        j3 = rem(rem(j - 1, size(u, 2)^2), size(u, 2)) + 1

        u_node = get_nodes_vars(u, equations, j1, j2, j3, k)
        flux_node1 = flux(u_node, 1, equations)
        flux_node2 = flux(u_node, 2, equations)
        flux_node3 = flux(u_node, 3, equations)

        @inbounds begin
            for ii in axes(u, 1)
                flux_arr1[ii, j1, j2, j3, k] = flux_node1[ii]
                flux_arr2[ii, j1, j2, j3, k] = flux_node2[ii]
                flux_arr3[ii, j1, j2, j3, k] = flux_node3[ii]
            end
        end
    end

    return nothing
end

# CUDA kernel for calculating weak form
function weak_form_kernel!(du, derivative_dhat, flux_arr1, flux_arr2, flux_arr3)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^3 && k <= size(du, 5))
        j1 = div(j - 1, size(du, 2)^2) + 1
        j2 = div(rem(j - 1, size(du, 2)^2), size(du, 2)) + 1
        j3 = rem(rem(j - 1, size(du, 2)^2), size(du, 2)) + 1

        @inbounds begin
            for ii in axes(du, 2)
                du[i, j1, j2, j3, k] += derivative_dhat[j1, ii] * flux_arr1[i, ii, j2, j3, k]
                du[i, j1, j2, j3, k] += derivative_dhat[j2, ii] * flux_arr2[i, j1, ii, j3, k]
                du[i, j1, j2, j3, k] += derivative_dhat[j3, ii] * flux_arr3[i, j1, j2, ii, k]
            end
        end
    end

    return nothing
end

# Calculate volume integral
function cuda_volume_integral!(du, u, mesh::TreeMesh{3},
    nonconservative_terms, equations,
    volume_integral::VolumeIntegralWeakForm, dg::DGSEM)

    derivative_dhat = CuArray{Float32}(dg.basis.derivative_dhat)
    flux_arr1 = similar(u)
    flux_arr2 = similar(u)
    flux_arr3 = similar(u)

    size_arr = CuArray{Float32}(undef, size(u, 2)^3, size(u, 5))

    flux_kernel = @cuda launch = false flux_kernel!(flux_arr1, flux_arr2, flux_arr3, u, equations, flux)
    flux_kernel(flux_arr1, flux_arr2, flux_arr3, u, equations; configurator_2d(flux_kernel, size_arr)...)

    size_arr = CuArray{Float32}(undef, size(du, 1), size(du, 2)^3, size(du, 5))

    weak_form_kernel = @cuda launch = false weak_form_kernel!(du, derivative_dhat, flux_arr1, flux_arr2, flux_arr3)
    weak_form_kernel(du, derivative_dhat, flux_arr1, flux_arr2, flux_arr3; configurator_3d(weak_form_kernel, size_arr)...)

    return nothing
end

# CUDA kernel for prolonging two interfaces in direction x, y, and z
function prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids, orientations)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(interfaces_u, 2) * size(interfaces_u, 3)^2 && k <= size(interfaces_u, 5))
        j1 = div(j - 1, size(interfaces_u, 3)^2) + 1
        j2 = div(rem(j - 1, size(interfaces_u, 3)^2), size(interfaces_u, 3)) + 1
        j3 = rem(rem(j - 1, size(interfaces_u, 3)^2), size(interfaces_u, 3)) + 1

        orientation = orientations[k]
        left_element = neighbor_ids[1, k]
        right_element = neighbor_ids[2, k]

        @inbounds begin
            interfaces_u[1, j1, j2, j3, k] = u[j1,
                isequal(orientation, 1)*size(u, 2)+isequal(orientation, 2)*j2+isequal(orientation, 3)*j2,
                isequal(orientation, 1)*j2+isequal(orientation, 2)*size(u, 2)+isequal(orientation, 3)*j3,
                isequal(orientation, 1)*j3+isequal(orientation, 2)*j3+isequal(orientation, 3)*size(u, 2),
                left_element]
            interfaces_u[2, j1, j2, j3, k] = u[j1,
                isequal(orientation, 1)*1+isequal(orientation, 2)*j2+isequal(orientation, 3)*j2,
                isequal(orientation, 1)*j2+isequal(orientation, 2)*1+isequal(orientation, 3)*j3,
                isequal(orientation, 1)*j3+isequal(orientation, 2)*j3+isequal(orientation, 3)*1,
                right_element]
        end
    end

    return nothing
end

# Prolong solution to interfaces
function cuda_prolong2interfaces!(u, mesh::TreeMesh{3}, cache)

    interfaces_u = CuArray{Float32}(cache.interfaces.u)
    neighbor_ids = CuArray{Int32}(cache.interfaces.neighbor_ids)
    orientations = CuArray{Int32}(cache.interfaces.orientations)

    size_arr = CuArray{Float32}(undef, size(interfaces_u, 2) * size(interfaces_u, 3)^2, size(interfaces_u, 5))

    prolong_interfaces_kernel = @cuda launch = false prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids, orientations)
    prolong_interfaces_kernel(interfaces_u, u, neighbor_ids, orientations; configurator_2d(prolong_interfaces_kernel, size_arr)...)

    cache.interfaces.u = interfaces_u  # Automatically copy back to CPU

    return nothing
end

# CUDA kernel for calculating surface fluxes 
function surface_flux_kernel!(surface_flux_arr, interfaces_u, orientations,
    equations::AbstractEquations{3}, surface_flux::Any)
    j2 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j3 = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (j2 <= size(surface_flux_arr, 3) && j3 <= size(surface_flux_arr, 4) && k <= size(surface_flux_arr, 5))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j2, j3, k)
        orientation = orientations[k]
        surface_flux_node = surface_flux(u_ll, u_rr, orientation, equations)

        @inbounds begin
            for j1j1 in axes(surface_flux_arr, 2)
                surface_flux_arr[1, j1j1, j2, j3, k] = surface_flux_node[j1j1]
            end
        end
    end

    return nothing
end

# CUDA kernel for setting interface fluxes on orientation 1, 2, and 3
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, neighbor_ids, orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_arr, 3)^2 && k <= size(surface_flux_arr, 5))
        j1 = div(j - 1, size(surface_flux_arr, 3)) + 1
        j2 = rem(j - 1, size(surface_flux_arr, 3)) + 1

        left_id = neighbor_ids[1, k]
        right_id = neighbor_ids[2, k]

        left_direction = 2 * orientations[k]
        right_direction = 2 * orientations[k] - 1

        @inbounds begin
            surface_flux_values[i, j1, j2, left_direction, left_id] = surface_flux_arr[1, i, j1, j2, k]
            surface_flux_values[i, j1, j2, right_direction, right_id] = surface_flux_arr[1, i, j1, j2, k]
        end
    end

    return nothing
end

# Calculate interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{3}, nonconservative_terms::False,
    equations, dg::DGSEM, cache)

    surface_flux = dg.surface_integral.surface_flux
    interfaces_u = CuArray{Float32}(cache.interfaces.u)
    neighbor_ids = CuArray{Int32}(cache.interfaces.neighbor_ids)
    orientations = CuArray{Int32}(cache.interfaces.orientations)
    surface_flux_arr = CuArray{Float32}(undef, 1, size(interfaces_u)[2:end]...)
    surface_flux_values = CuArray{Float32}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float32}(undef, size(interfaces_u, 3), size(interfaces_u, 4), size(interfaces_u, 5))

    surface_flux_kernel = @cuda launch = false surface_flux_kernel!(surface_flux_arr, interfaces_u, orientations, equations, surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, orientations, equations, surface_flux; configurator_3d(surface_flux_kernel, size_arr)...)

    size_arr = CuArray{Float32}(undef, size(surface_flux_values, 1), size(interfaces_u, 3)^2, size(interfaces_u, 5))

    interface_flux_kernel = @cuda launch = false interface_flux_kernel!(surface_flux_values, surface_flux_arr, neighbor_ids, orientations)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, neighbor_ids, orientations; configurator_3d(interface_flux_kernel, size_arr)...)

    cache.elements.surface_flux_values = surface_flux_values # Automatically copy back to CPU

    return nothing
end

# CUDA kernel for calculating surface integrals along axis x, y, and z
function surface_integral_kernel!(du, factor_arr, surface_flux_values)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 5))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds begin
            du[i, 1, j1, j2, k] -= surface_flux_values[i, j1, j2, 1, k] * factor_arr[1]
            du[i, size(du, 2), j1, j2, k] += surface_flux_values[i, j1, j2, 2, k] * factor_arr[2]
            du[i, j1, 1, j2, k] -= surface_flux_values[i, j1, j2, 3, k] * factor_arr[1]
            du[i, j1, size(du, 2), j2, k] += surface_flux_values[i, j1, j2, 4, k] * factor_arr[2]
            du[i, j1, j2, 1, k] -= surface_flux_values[i, j1, j2, 5, k] * factor_arr[1]
            du[i, j1, j2, size(du, 2), k] += surface_flux_values[i, j1, j2, 6, k] * factor_arr[2]
        end
    end

    return nothing
end

# CUDA kernel for calculating surface integrals along axis x
function surface_integral_kernel1!(du, factor_arr, surface_flux_values)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 5))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds begin
            du[i, 1, j1, j2, k] -= surface_flux_values[i, j1, j2, 1, k] * factor_arr[1]
            du[i, size(du, 2), j1, j2, k] += surface_flux_values[i, j1, j2, 2, k] * factor_arr[2]
        end
    end

    return nothing
end

# CUDA kernel for calculating surface integrals along axis y
function surface_integral_kernel2!(du, factor_arr, surface_flux_values)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 5))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds begin
            du[i, j1, 1, j2, k] -= surface_flux_values[i, j1, j2, 3, k] * factor_arr[1]
            du[i, j1, size(du, 2), j2, k] += surface_flux_values[i, j1, j2, 4, k] * factor_arr[2]
        end
    end

    return nothing
end

# CUDA kernel for calculating surface integrals along axis z
function surface_integral_kernel3!(du, factor_arr, surface_flux_values)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 5))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds begin
            du[i, j1, j2, 1, k] -= surface_flux_values[i, j1, j2, 5, k] * factor_arr[1]
            du[i, j1, j2, size(du, 2), k] += surface_flux_values[i, j1, j2, 6, k] * factor_arr[2]
        end
    end

    return nothing
end

# Calculate surface integrals
function cuda_surface_integral!(du, mesh::TreeMesh{3}, dg::DGSEM, cache) # surface_integral

    factor_arr = CuArray{Float32}([dg.basis.boundary_interpolation[1, 1], dg.basis.boundary_interpolation[size(du, 2), 2]])
    surface_flux_values = CuArray{Float32}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float32}(undef, size(du, 1), size(du, 2)^2, size(du, 5))

    surface_integral_kernel1 = @cuda launch = false surface_integral_kernel1!(du, factor_arr, surface_flux_values)
    surface_integral_kernel1(du, factor_arr, surface_flux_values; configurator_3d(surface_integral_kernel1, size_arr)...)

    surface_integral_kernel2 = @cuda launch = false surface_integral_kernel2!(du, factor_arr, surface_flux_values)
    surface_integral_kernel2(du, factor_arr, surface_flux_values; configurator_3d(surface_integral_kernel2, size_arr)...)

    surface_integral_kernel3 = @cuda launch = false surface_integral_kernel3!(du, factor_arr, surface_flux_values)
    surface_integral_kernel3(du, factor_arr, surface_flux_values; configurator_3d(surface_integral_kernel3, size_arr)...)

    return nothing
end

# Inside `rhs!()` raw implementation
#################################################################################
du, u = copy_to_gpu!(du, u)

cuda_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver)

cuda_prolong2interfaces!(u, mesh, cache)

cuda_interface_flux!(
    mesh, have_nonconservative_terms(equations),
    equations, solver, cache,)

cuda_surface_integral!(du, mesh, solver, cache)

# For tests
#################################################################################
#= reset_du!(du, solver, cache)

calc_volume_integral!(
    du, u, mesh,
    have_nonconservative_terms(equations), equations,
    solver.volume_integral, solver, cache)

prolong2interfaces!(
    cache, u, mesh, equations, solver.surface_integral, solver)

calc_interface_flux!(
    cache.elements.surface_flux_values, mesh,
    have_nonconservative_terms(equations), equations,
    solver.surface_integral, solver, cache)

calc_surface_integral!(
    du, u, mesh, equations, solver.surface_integral, solver, cache) =#