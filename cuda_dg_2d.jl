# Remove it after first run to avoid recompilation
#= include("header.jl") =#

# Use the target test header file
#= include("test/linear_scalar_advection_2d.jl") =#
include("test/compressible_euler_2d.jl")

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

# CUDA kernel for calculating fluxes along normal direction 1 and 2
function flux_kernel!(flux_arr1, flux_arr2, u, equations::AbstractEquations{2}, flux::Function)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(u, 2)^2 && k <= size(u, 4))
        j1 = div(j - 1, size(u, 2)) + 1
        j2 = rem(j - 1, size(u, 2)) + 1

        u_node = get_nodes_vars(u, equations, j1, j2, k)
        flux_node1 = flux(u_node, 1, equations)
        flux_node2 = flux(u_node, 2, equations)

        @inbounds begin
            for ii in axes(u, 1)
                flux_arr1[ii, j1, j2, k] = flux_node1[ii]
                flux_arr2[ii, j1, j2, k] = flux_node2[ii]
            end
        end
    end

    return nothing
end

# CUDA kernel for calculating weak form
function weak_form_kernel!(du, derivative_dhat, flux_arr1, flux_arr2)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds begin
            for ii in axes(du, 2)
                du[i, j1, j2, k] += derivative_dhat[j1, ii] * flux_arr1[i, ii, j2, k]
                du[i, j1, j2, k] += derivative_dhat[j2, ii] * flux_arr2[i, j1, ii, k]
            end
        end
    end

    return nothing
end

# Calculate volume integral
function cuda_volume_integral!(du, u, mesh::TreeMesh{2},
    nonconservative_terms, equations,
    volume_integral::VolumeIntegralWeakForm, dg::DGSEM)

    derivative_dhat = CuArray{Float32}(dg.basis.derivative_dhat)
    flux_arr1 = similar(u)
    flux_arr2 = similar(u)

    size_arr = CuArray{Float32}(undef, size(u, 2)^2, size(u, 4))

    flux_kernel = @cuda launch = false flux_kernel!(flux_arr1, flux_arr2, u, equations, flux)
    flux_kernel(flux_arr1, flux_arr2, u, equations; configurator_2d(flux_kernel, size_arr)...)

    size_arr = CuArray{Float32}(undef, size(du, 1), size(du, 2)^2, size(du, 4))

    weak_form_kernel = @cuda launch = false weak_form_kernel!(du, derivative_dhat, flux_arr1, flux_arr2)
    weak_form_kernel(du, derivative_dhat, flux_arr1, flux_arr2; configurator_3d(weak_form_kernel, size_arr)...)

    return nothing
end

# CUDA kernel for prolonging two interfaces in direction x and y
function prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids, orientations)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(interfaces_u, 2) * size(interfaces_u, 3) && k <= size(interfaces_u, 4))
        j1 = div(j - 1, size(interfaces_u, 3)) + 1
        j2 = rem(j - 1, size(interfaces_u, 3)) + 1

        orientation = orientations[k]
        left_element = neighbor_ids[1, k]
        right_element = neighbor_ids[2, k]

        @inbounds begin
            interfaces_u[1, j1, j2, k] = u[j1,
                isequal(orientation, 1)*j2+isequal(orientation, 2)*size(u, 2),
                isequal(orientation, 1)*size(u, 2)+isequal(orientation, 2)*j2,
                left_element]
            interfaces_u[2, j1, j2, k] = u[j1,
                isequal(orientation, 1)*j2+isequal(orientation, 2)*1,
                isequal(orientation, 1)*1+isequal(orientation, 2)*j2,
                right_element]
        end
    end

    return nothing
end

# Prolong solution to interfaces
function cuda_prolong2interfaces!(u, mesh::TreeMesh{2}, cache)

    interfaces_u = CuArray{Float32}(cache.interfaces.u)
    neighbor_ids = CuArray{Int32}(cache.interfaces.neighbor_ids)
    orientations = CuArray{Int32}(cache.interfaces.orientations)

    size_arr = CuArray{Float32}(undef, size(interfaces_u, 2) * size(interfaces_u, 3), size(interfaces_u, 4))

    prolong_interfaces_kernel = @cuda launch = false prolong_interfaces_kernel!(interfaces_u, u, neighbor_ids, orientations)
    prolong_interfaces_kernel(interfaces_u, u, neighbor_ids, orientations; configurator_2d(prolong_interfaces_kernel, size_arr)...)

    cache.interfaces.u = interfaces_u  # Automatically copy back to CPU

    return nothing
end

# CUDA kernel for calculating surface fluxes 
function surface_flux_kernel!(surface_flux_arr, interfaces_u, orientations,
    equations::AbstractEquations{2}, surface_flux::Any)
    j2 = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j2 <= size(surface_flux_arr, 3) && k <= size(surface_flux_arr, 4))
        u_ll, u_rr = get_surface_node_vars(interfaces_u, equations, j2, k)
        orientation = orientations[k]
        surface_flux_node = surface_flux(u_ll, u_rr, orientation, equations)

        @inbounds begin
            for j1j1 in axes(surface_flux_arr, 2)
                surface_flux_arr[1, j1j1, j2, k] = surface_flux_node[j1j1]
            end
        end
    end

    return nothing
end

# CUDA kernel for setting interface fluxes on orientation 1 and 2
function interface_flux_kernel!(surface_flux_values, surface_flux_arr, neighbor_ids, orientations)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(surface_flux_values, 1) && j <= size(surface_flux_arr, 3) && k <= size(surface_flux_arr, 4))
        left_id = neighbor_ids[1, k]
        right_id = neighbor_ids[2, k]

        left_direction = 2 * orientations[k]
        right_direction = 2 * orientations[k] - 1

        @inbounds begin
            surface_flux_values[i, j, left_direction, left_id] = surface_flux_arr[1, i, j, k]
            surface_flux_values[i, j, right_direction, right_id] = surface_flux_arr[1, i, j, k]
        end
    end

    return nothing
end

# Calculate interface fluxes
function cuda_interface_flux!(mesh::TreeMesh{2}, nonconservative_terms::False,
    equations, dg::DGSEM, cache)

    surface_flux = dg.surface_integral.surface_flux
    interfaces_u = CuArray{Float32}(cache.interfaces.u)
    neighbor_ids = CuArray{Int32}(cache.interfaces.neighbor_ids)
    orientations = CuArray{Int32}(cache.interfaces.orientations)
    surface_flux_arr = CuArray{Float32}(undef, 1, size(interfaces_u)[2:end]...)
    surface_flux_values = CuArray{Float32}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float32}(undef, size(interfaces_u, 3), size(interfaces_u, 4))

    surface_flux_kernel = @cuda launch = false surface_flux_kernel!(surface_flux_arr, interfaces_u, orientations, equations, surface_flux)
    surface_flux_kernel(surface_flux_arr, interfaces_u, orientations, equations, surface_flux; configurator_2d(surface_flux_kernel, size_arr)...)

    size_arr = CuArray{Float32}(undef, size(surface_flux_values, 1), size(interfaces_u, 3), size(interfaces_u, 4))

    interface_flux_kernel = @cuda launch = false interface_flux_kernel!(surface_flux_values, surface_flux_arr, neighbor_ids, orientations)
    interface_flux_kernel(surface_flux_values, surface_flux_arr, neighbor_ids, orientations; configurator_3d(interface_flux_kernel, size_arr)...)

    cache.elements.surface_flux_values = surface_flux_values # Automatically copy back to CPU

    return nothing
end

# CUDA kernel for calculating surface integrals along axis x and y
function surface_integral_kernel!(du, factor_arr, surface_flux_values)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 4))
        @inbounds begin
            du[i, 1, j, k] -= surface_flux_values[i, j, 1, k] * factor_arr[1]
            du[i, size(du, 2), j, k] += surface_flux_values[i, j, 2, k] * factor_arr[2]
            du[i, j, 1, k] -= surface_flux_values[i, j, 3, k] * factor_arr[1]
            du[i, j, size(du, 2), k] += surface_flux_values[i, j, 4, k] * factor_arr[2]
        end
    end

    return nothing
end

# CUDA kernel for calculating surface integrals along axis x
function surface_integral_kernel1!(du, factor_arr, surface_flux_values)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 4))
        @inbounds begin
            du[i, 1, j, k] -= surface_flux_values[i, j, 1, k] * factor_arr[1]
            du[i, size(du, 2), j, k] += surface_flux_values[i, j, 2, k] * factor_arr[2]
        end
    end

    return nothing
end

# CUDA kernel for calculating surface integrals along axis y
function surface_integral_kernel2!(du, factor_arr, surface_flux_values)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2) && k <= size(du, 4))
        @inbounds begin
            du[i, j, 1, k] -= surface_flux_values[i, j, 3, k] * factor_arr[1]
            du[i, j, size(du, 2), k] += surface_flux_values[i, j, 4, k] * factor_arr[2]
        end
    end

    return nothing
end

# Calculate surface integrals
function cuda_surface_integral!(du, mesh::TreeMesh{2}, dg::DGSEM, cache) # surface_integral

    factor_arr = CuArray{Float32}([dg.basis.boundary_interpolation[1, 1], dg.basis.boundary_interpolation[size(du, 2), 2]])
    surface_flux_values = CuArray{Float32}(cache.elements.surface_flux_values)

    size_arr = CuArray{Float32}(undef, size(du, 1), size(du, 2), size(du, 4))

    surface_integral_kernel1 = @cuda launch = false surface_integral_kernel1!(du, factor_arr, surface_flux_values)
    surface_integral_kernel1(du, factor_arr, surface_flux_values; configurator_3d(surface_integral_kernel1, size_arr)...)

    surface_integral_kernel2 = @cuda launch = false surface_integral_kernel2!(du, factor_arr, surface_flux_values)
    surface_integral_kernel2(du, factor_arr, surface_flux_values; configurator_3d(surface_integral_kernel2, size_arr)...)

    return nothing
end

# CUDA kernel for applying inverse Jacobian 
function jacobian_kernel!(du, inverse_jacobian)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

    if (i <= size(du, 1) && j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        @inbounds du[i, j1, j2, k] *= -inverse_jacobian[k]
    end

    return nothing
end

# Apply Jacobian from mapping to reference element
function cuda_jacobian!(du, mesh::TreeMesh{2}, cache)

    inverse_jacobian = CuArray{Float32}(cache.elements.inverse_jacobian)

    size_arr = CuArray{Float32}(undef, size(du, 1), size(du, 2)^2, size(du, 4))

    jacobian_kernel = @cuda launch = false jacobian_kernel!(du, inverse_jacobian)
    jacobian_kernel(du, inverse_jacobian; configurator_3d(jacobian_kernel, size_arr)...)

    return nothing
end

# CUDA kernel for calculating source terms
function source_terms_kernel!(du, u, node_coordinates, t, equations::AbstractEquations{2}, source_terms::Function)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    k = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (j <= size(du, 2)^2 && k <= size(du, 4))
        j1 = div(j - 1, size(du, 2)) + 1
        j2 = rem(j - 1, size(du, 2)) + 1

        u_local = get_nodes_vars(u, equations, j1, j2, k)
        x_local = get_node_coords(node_coordinates, equations, j1, j2, k)
        source_terms_node = source_terms(u_local, x_local, t, equations)

        @inbounds begin
            for ii in axes(du, 1)
                du[ii, j1, j2, k] += source_terms_node[ii]
            end
        end
    end

    return nothing
end

# Calculate source terms               
function cuda_sources!(du, u, t, source_terms::Nothing,
    equations::AbstractEquations{2}, cache)

    return nothing
end

# Calculate source terms 
function cuda_sources!(du, u, t, source_terms,
    equations::AbstractEquations{2}, cache)

    node_coordinates = CuArray{Float32}(cache.elements.node_coordinates)
    size_arr = CuArray{Float32}(undef, size(u, 2)^2, size(u, 4))

    source_terms_kernel = @cuda launch = false source_terms_kernel!(du, u, node_coordinates, t, equations, source_terms)
    source_terms_kernel(du, u, node_coordinates, t, equations, source_terms; configurator_2d(source_terms_kernel, size_arr)...)

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

cuda_jacobian!(du, mesh, cache)

cuda_sources!(du, u, t,
    source_terms, equations, cache)

du, u = copy_to_cpu!(du, u)

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
    du, u, mesh, equations, solver.surface_integral, solver, cache)

apply_jacobian!(du, mesh, equations, solver, cache)

calc_sources!(du, u, t,
    source_terms, equations, solver, cache) =#