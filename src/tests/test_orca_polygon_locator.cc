/*
 * (C) Copyright 2021- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <numeric>
#include <sstream>

#include "atlas/functionspace/NodeColumns.h"
#include "atlas/grid.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/util/Config.h"
#include "atlas/output/Gmsh.h"

#include "atlas/util/Geometry.h"
#include "atlas/util/LonLatMicroDeg.h"
#include "atlas/util/PeriodicTransform.h"

#include "atlas/util/PolygonLocator.h"
#include "atlas/util/PolygonXY.h"

#include "atlas-orca/grid/OrcaGrid.h"

#include "tests/AtlasTestEnvironment.h"

using Grid   = atlas::Grid;
using Config = atlas::util::Config;

namespace atlas {
namespace test {

//-----------------------------------------------------------------------------

void build_periodic_boundaries( Mesh& mesh );  // definition below

//-----------------------------------------------------------------------------


// CASE( "test orca polygon locator" ) {
// 
//     auto gridnames = std::vector<std::string>{
//         "ORCA2_T",   //
//         "eORCA1_T",  //
//         "eORCA025_T",  //
//         // "eORCA12_T",  //
//     };
// 
//     std::string grid_resource = eckit::Resource<std::string>( "--grid", "" );
//     if ( not grid_resource.empty() ) {
//         gridnames = {grid_resource};
//     }
// 
//     for ( auto gridname : gridnames ) {
//         SECTION( gridname ) {
//             OrcaGrid grid      = Grid( gridname );
//             grid::Partitioner partitioner("equal_regions", atlas::mpi::size());
//             auto meshgenerator = MeshGenerator{"orca"};
//             auto mesh          = meshgenerator.generate( grid, partitioner );
//             REQUIRE( mesh.grid() );
//             EXPECT( mesh.grid().name() == gridname );
//             atlas::util::ListPolygonXY polygon_list(mesh.polygons());
//             //mesh.polygon(0).outputPythonScript("polygon.py",
//             //                                   Config("nodes", false)("coordinates", "xy"));
//             atlas::util::PolygonLocator locator(polygon_list, mesh.projection());
// 
//             // fails on ORCA2 because partition polygon is not connected space
//             // (ORCA2 grids have a cut out area for the mediterranean)
//             idx_t part;
//             part = locator({82.0, 10.0});
//             ATLAS_DEBUG_VAR( part );
//             // would fail on ORCA025 because xy coordinate system doesn't cover
//             // 0-360. A fix for this has been added to the locator in atlas 0.32.0.
//             part = locator({-173.767, -61.1718});
//             ATLAS_DEBUG_VAR( part );
// 
//         }
//     }
// }

CASE( "test haloExchange " ) {
    auto gridnames = std::vector<std::string>{
        "ORCA2_T",   //
        //"eORCA1_T",  //
        //"eORCA025_T",  //
    };
    for ( auto gridname : gridnames ) {
        SECTION( gridname ) {
            int64_t halo = 0;
            auto grid = Grid(gridname);
            auto meshgen_config = grid.meshgenerator() | option::halo(halo);
            atlas::MeshGenerator meshgen(meshgen_config);
            auto partitioner_config = grid.partitioner();
            partitioner_config.set("type", "serial");
            auto partitioner = grid::Partitioner(partitioner_config);
            auto mesh = meshgen.generate(grid, partitioner);
            //test::build_periodic_boundaries( mesh );
            REQUIRE( mesh.grid() );
            EXPECT( mesh.grid().name() == gridname );
            idx_t count{0};

            const auto remote_idxs = array::make_indexview<idx_t, 1>(
                                      mesh.nodes().remote_index());
            if (mpi::rank() == 0) std::cout << mpi::rank() << " test_orca_polygon_locator before function space create remote_idxs(0) " << remote_idxs(0) << std::endl;
            functionspace::NodeColumns fs{mesh};
            if (mpi::rank() == 0) std::cout << mpi::rank() << " test_orca_polygon_locator after function space create remote_idxs(0) " << remote_idxs(0) << std::endl;
            Field field   = fs.createField<double>( option::name( "unswapped ghosts" ) );
            if (mpi::rank() == 0) std::cout << mpi::rank() << " test_orca_polygon_locator after create field remote_idxs(0) " << remote_idxs(0) << std::endl;
            auto f        = array::make_view<double, 1>( field );
            const auto ghosts = atlas::array::make_view<int32_t, 1>(
                                  mesh.nodes().ghost());
            if (mpi::rank() == 0) std::cout << mpi::rank() << " test_orca_polygon_locator after create ghost view remote_idxs(0) " << remote_idxs(0) << std::endl;
            for ( idx_t jnode = 0; jnode < mesh.nodes().size(); ++jnode ) {
                if (ghosts(jnode)) {
                    f( jnode ) = 1;
                } else {
                    f( jnode ) = 0;
                }
            }
            if (mpi::rank() == 0) std::cout << mpi::rank() << " test_orca_polygon_locator after iterate over field and ghost view remote_idxs(0) " << remote_idxs(0) << std::endl;

            fs.haloExchange(field);

            Field field2   = fs.createField<double>( option::name( "ghost points to itself" ) );
            auto f_ghost_recurrent = array::make_view<double, 1>( field2 );
            Field field3   = fs.createField<double>( option::name( "topology BC" ) );
            auto f_bc      = array::make_view<double, 1>( field3 );
            Field field4   = fs.createField<double>( option::name( "topology PERIODIC" ) );
            auto f_periodic = array::make_view<double, 1>( field4 );
            const auto parts       = atlas::array::make_view<int32_t, 1>(
                                      mesh.nodes().partition());
            const auto mypart = atlas::mpi::rank();
            auto flags_v = array::make_view<int, 1>( mesh.nodes().flags() );

            for ( idx_t jnode = 0; jnode < mesh.nodes().size(); ++jnode ) {
                f_ghost_recurrent( jnode ) = 0;
                f_bc( jnode ) = 0;
                f_periodic( jnode ) = 0;
                if (ghosts(jnode) &&
                    (parts(jnode) == mypart) &&
                    remote_idxs(jnode) == jnode) {
                    f_ghost_recurrent( jnode ) = 1;
                }
                auto flags = util::Topology::view( flags_v( jnode ) );
                if ( flags.check_all( util::Topology::BC ) ) {
                    f_bc( jnode ) = 1;
                }
                if ( flags.check_all( util::Topology::PERIODIC ) ) {
                    f_periodic( jnode ) = 1;
                }
                if (f( jnode )) {
                  ++count;
                }
            }
            if ( count != 0 ) {
                Log::info() << "To diagnose problem, uncomment mesh writing here: " << Here() << std::endl;
                output::Gmsh gmsh(std::string("haloExchange_")+gridname+".msh",Config("coordinates","ij")|Config("info",true));
                gmsh.write(mesh);
                gmsh.write(field);
                gmsh.write(field2);
                gmsh.write(field3);
                gmsh.write(field4);
            }
            EXPECT_EQ( count, 0 );
        }
    }
}

void build_periodic_boundaries( Mesh& mesh ) {
    using util::LonLatMicroDeg;
    using util::PeriodicTransform;
    using util::Topology;

    ATLAS_TRACE();
    bool periodic = false;
    mesh.metadata().get( "periodic", periodic );

    auto mpi_size = mpi::size();
    auto mypart   = mpi::rank();

    if (periodic) {
      mesh.metadata().set( "periodic", true );
      return;
    }

    mesh::Nodes& nodes = mesh.nodes();

    auto flags_v = array::make_view<int, 1>( nodes.flags() );
    auto ridx    = array::make_indexview<idx_t, 1>( nodes.remote_index() );
    auto part    = array::make_view<int, 1>( nodes.partition() );
    auto ghost   = array::make_view<int, 1>( nodes.ghost() );

    int nb_nodes = nodes.size();

    auto xy = array::make_view<double, 2>( nodes.xy() );
    auto ij = array::make_view<idx_t, 2>( nodes.field( "ij" ) );

    // Identify my master and periodic_ghost nodes on own partition
    // master nodes are at x=0,  periodic_ghost nodes are at x=2pi
    std::map<uid_t, int> master_lookup;
    std::map<uid_t, int> periodic_ghost_lookup;
    std::vector<int> master_nodes;
    master_nodes.reserve( 3 * nb_nodes );
    std::vector<int> periodic_ghost_nodes;
    periodic_ghost_nodes.reserve( 3 * nb_nodes );

    auto collect_periodic_ghost_and_master_nodes = [&]() {
        for ( idx_t jnode = 0; jnode < nodes.size(); ++jnode ) {
            auto flags = Topology::view( flags_v( jnode ) );

            if ( flags.check_all( Topology::BC | Topology::WEST ) ) {
                flags.set( Topology::PERIODIC );
                if ( part( jnode ) == mypart ) {
                    LonLatMicroDeg ll( xy( jnode, XX ), xy( jnode, YY ) );
                    master_lookup[ll.unique()] = jnode;
                    master_nodes.push_back( ll.lon() );
                    master_nodes.push_back( ll.lat() );
                    master_nodes.push_back( jnode );
                }
                //Log::info() << "master " << jnode << "  " << PointXY{ij( jnode, 0 ), ij( jnode, 1 )} << std::endl;
            }
            else if ( flags.check( Topology::BC | Topology::EAST ) ) {
                flags.set( Topology::PERIODIC | Topology::GHOST );
                ghost( jnode ) = 1;
                LonLatMicroDeg ll( xy( jnode, XX ), xy( jnode, YY ) );
                periodic_ghost_lookup[ll.unique()] = jnode;
                periodic_ghost_nodes.push_back( ll.lon() );
                periodic_ghost_nodes.push_back( ll.lat() );
                periodic_ghost_nodes.push_back( jnode );
                ridx( jnode ) = -1;
                Log::info() << "periodic_ghost " << jnode << "  " << atlas::orca::PointIJ{ij( jnode, 0 ), ij( jnode, 1 )} << std::endl;
            }
        }
    };

    auto collect_periodic_ghost_and_master_nodes_halo = [&]() {
        auto master_glb_idx = array::make_view<gidx_t, 1>( mesh.nodes().field( "master_global_index" ) );
        auto glb_idx        = array::make_view<gidx_t, 1>( mesh.nodes().global_index() );
        for ( idx_t jnode = 0; jnode < nodes.size(); ++jnode ) {
            auto flags = Topology::view( flags_v( jnode ) );
            if ( flags.check( Topology::PERIODIC ) ) {
                if ( master_glb_idx( jnode ) == glb_idx( jnode ) ) {
                    if ( part( jnode ) == mypart ) {
                        master_lookup[master_glb_idx( jnode )] = jnode;
                        master_nodes.push_back( master_glb_idx( jnode ) );
                        master_nodes.push_back( 0 );
                        master_nodes.push_back( jnode );
                        //Log::info() << "master " << jnode << "  " << PointXY{ ij(jnode,0), ij(jnode,1)} << std::endl;
                    }
                }
                else {
                    periodic_ghost_lookup[master_glb_idx( jnode )] = jnode;
                    periodic_ghost_nodes.push_back( master_glb_idx( jnode ) );
                    periodic_ghost_nodes.push_back( 0 );
                    periodic_ghost_nodes.push_back( jnode );
                    ridx( jnode ) = -1;
                    //Log::info() << "periodic_ghost " << jnode << "  " << PointXY{ ij(jnode,0), ij(jnode,1)} << std::endl;
                }
            }
        }
    };

    bool require_PeriodicTransform;
    if ( mesh.nodes().has_field( "master_global_index" ) ) {
        collect_periodic_ghost_and_master_nodes_halo();
        require_PeriodicTransform = false;
    }
    else {
        collect_periodic_ghost_and_master_nodes();
        require_PeriodicTransform = true;
    }
    std::vector<std::vector<int>> found_master( mpi_size );
    std::vector<std::vector<int>> send_periodic_ghost_idx( mpi_size );

    // Find masters on other tasks to send to me
    {
        int sendcnt = periodic_ghost_nodes.size();
        std::vector<int> recvcounts( mpi_size );

        ATLAS_TRACE_MPI( ALLGATHER ) { mpi::comm().allGather( sendcnt, recvcounts.begin(), recvcounts.end() ); }

        std::vector<int> recvdispls( mpi_size );
        recvdispls[0] = 0;
        int recvcnt   = recvcounts[0];
        for ( idx_t jproc = 1; jproc < mpi_size; ++jproc ) {
            recvdispls[jproc] = recvdispls[jproc - 1] + recvcounts[jproc - 1];
            recvcnt += recvcounts[jproc];
        }
        std::vector<int> recvbuf( recvcnt );

        ATLAS_TRACE_MPI( ALLGATHER ) {
            mpi::comm().allGatherv( periodic_ghost_nodes.begin(), periodic_ghost_nodes.end(), recvbuf.begin(), recvcounts.data(),
                                    recvdispls.data() );
        }

        PeriodicTransform transform;
        for ( idx_t jproc = 0; jproc < mpi_size; ++jproc ) {
            array::LocalView<int, 2> recv_periodic_ghost( recvbuf.data() + recvdispls[jproc],
                                                 array::make_shape( recvcounts[jproc] / 3, 3 ) );
            for ( idx_t jnode = 0; jnode < recv_periodic_ghost.shape( 0 ); ++jnode ) {
                uid_t periodic_ghost_uid;
                if ( require_PeriodicTransform ) {
                    LonLatMicroDeg periodic_ghost( recv_periodic_ghost( jnode, LON ), recv_periodic_ghost( jnode, LAT ) );
                    transform( periodic_ghost, -1 );
                    uid_t periodic_ghost_uid = periodic_ghost.unique();
                }
                else {
                    periodic_ghost_uid = recv_periodic_ghost( jnode, 0 );
                }
                int debug_periodic_ghost_idx  = recv_periodic_ghost( jnode, 2 );
                std::stringstream debug_out;
                debug_out << "periodic_ghost index = " << debug_periodic_ghost_idx
                          << " at " << (PointXY{ ij(debug_periodic_ghost_idx,0), ij(debug_periodic_ghost_idx,1)})
                          << "  uid = " << periodic_ghost_uid;
                if ( master_lookup.count( periodic_ghost_uid ) ) {
                    int master_idx = master_lookup[periodic_ghost_uid];
                    int periodic_ghost_idx  = recv_periodic_ghost( jnode, 2 );
                    debug_out << " found in master " << (PointXY{ij(master_idx,0),ij(master_idx,1)}) << std::endl;
                    ATLAS_DEBUG( debug_out.str() );
                    found_master[jproc].push_back( master_idx );
                    send_periodic_ghost_idx[jproc].push_back( periodic_ghost_idx );
                }
            }
        }
    }

    // Fill in data to communicate
    std::vector<std::vector<int>> recv_periodic_ghost_idx( mpi_size );
    std::vector<std::vector<int>> send_master_part( mpi_size );
    std::vector<std::vector<int>> recv_master_part( mpi_size );
    std::vector<std::vector<int>> send_master_ridx( mpi_size );
    std::vector<std::vector<int>> recv_master_ridx( mpi_size );

    {
        for ( idx_t jproc = 0; jproc < mpi_size; ++jproc ) {
            idx_t nb_found_master = static_cast<idx_t>( found_master[jproc].size() );
            send_master_part[jproc].resize( nb_found_master );
            send_master_ridx[jproc].resize( nb_found_master );
            for ( idx_t jnode = 0; jnode < nb_found_master; ++jnode ) {
                int loc_idx                    = found_master[jproc][jnode];
                send_master_part[jproc][jnode] = part( loc_idx );
                send_master_ridx[jproc][jnode] = loc_idx;
            }
        }
    }

    // Communicate
    ATLAS_TRACE_MPI( ALLTOALL ) {
        mpi::comm().allToAll( send_periodic_ghost_idx, recv_periodic_ghost_idx );
        mpi::comm().allToAll( send_master_part, recv_master_part );
        mpi::comm().allToAll( send_master_ridx, recv_master_ridx );
    }

    // Fill in periodic
    for ( idx_t jproc = 0; jproc < mpi_size; ++jproc ) {
        idx_t nb_recv = static_cast<idx_t>( recv_periodic_ghost_idx[jproc].size() );
        for ( idx_t jnode = 0; jnode < nb_recv; ++jnode ) {
            idx_t periodic_ghost_idx   = recv_periodic_ghost_idx[jproc][jnode];
            part( periodic_ghost_idx ) = recv_master_part[jproc][jnode];
            ridx( periodic_ghost_idx ) = recv_master_ridx[jproc][jnode];
        }
    }
    mesh.metadata().set( "periodic", true );
}

}  // namespace test
}  // namespace atlas

int main( int argc, char** argv ) {
    return atlas::test::run( argc, argv );
}
