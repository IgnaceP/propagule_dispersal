#=
tin interpolator:
- Julia version: 
- Author: ignace
- Date: 2026-11-23
=#
using NearestNeighbors
using Base.Threads

function barycentric_coordinates(A,B,C,P)

    D = (B[2]- C[2])*(A[1]-C[1]) + (C[1]-B[1])*(A[2]-C[2])
    alpha = ((B[2]-C[2])*(P[1]-C[1]) + (C[1]-B[1])*(P[2]-C[2])) / D
    beta = ((C[2]-A[2])*(P[1]-C[1]) + (A[1]-C[1])*(P[2]-C[2])) / D
    gamma = 1 - alpha - beta

    return alpha, beta, gamma
end

function tin_interpolator(pois::Matrix,
    triangles_xy::Matrix, triangles_connectivity::Matrix, triangle_values::Matrix,
    vertex_to_triangles::Dict)

    # number of triangles to check
    k = size(triangles_xy, 1)

    # build a 2 x n matrix of triangle vertices
    # this is essential for KDTree
    xy_matrix = transpose(triangles_xy)

    # build a KD tree
    kdtree = KDTree(xy_matrix; leafsize = 25)

    # initialize vector to store interpolated values
    pois_values = fill(Float32(NaN), size(pois)[1], size(triangle_values)[2])

    Threads.@threads for poi_i in 1:size(pois)[1]
    #for poi_i in 1:size(pois)[1]

        # get the point of interest
        poi = @view pois[poi_i,:]

        if isnan(poi[1]) || isnan(poi[2])
            continue
        end

        # find the k-nearest vertices to the point of interest
        idx, dists = knn(kdtree, poi, k, true)

        # initialize variables
        triangle_found = false

        # loop over the number of triangles, ordered by vertex distance to the point of interest
        for neigh in idx

            # Extract row indices from CartesianIndex objects
            #connectivity_rows = unique(i[1] for i in findall(x -> any(x .== neigh), triangles_connectivity))
            # Use the row indices to get the rows of triangles_connectivity
            #triangles = @view triangles_connectivity[connectivity_rows, :]

            # loop over all triangles from which neigh is a vertex of
            for triangle_idx in vertex_to_triangles[neigh]
                triangle = @view triangles_connectivity[triangle_idx, :]

                # get coordinates of all triangle vertices
                A = @view triangles_xy[triangle[1],:]
                B = @view triangles_xy[triangle[2],:]
                C = @view triangles_xy[triangle[3],:]

                # calculate barycentric coordinates of the point of interest with respect to the triangle
                alpha,beta,gamma=barycentric_coordinates(A, B, C, poi)

                # check if the point of interest is inside the triangle
                if alpha >= -1e-12 && beta >= -1e-12 && gamma >= -1e-12

                    for (val_i,val) in enumerate(eachcol(triangle_values))
                        # compute the interpolated value using barycentric coordinates
                        pois_values[poi_i,val_i] =      alpha * val[triangle[1]] +
                                                        beta * val[triangle[2]] +
                                                        gamma * val[triangle[3]]

                    end

                    triangle_found = true

                    break
                end
            end

            if triangle_found
                break
            end
        end

    end

    return pois_values
end