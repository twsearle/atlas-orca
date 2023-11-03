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
#include "atlas-orca/meshgenerator/OrcaMeshGenerator.h"
#include "atlas/util/Config.h"
#include "atlas/util/function/VortexRollup.h"
#include "atlas/output/Gmsh.h"

#include "atlas/util/Geometry.h"
#include "atlas/util/LonLatMicroDeg.h"
#include "atlas/util/PeriodicTransform.h"

#include "atlas-orca/grid/OrcaGrid.h"
#include "atlas-orca/util/PointIJ.h"

#include "tests/AtlasTestEnvironment.h"

using Grid   = atlas::Grid;
using Config = atlas::util::Config;

namespace atlas {
namespace test {

//-----------------------------------------------------------------------------

void build_remote_index(Mesh& mesh);

//-----------------------------------------------------------------------------


CASE( "test haloExchange " ) {
    auto gridnames = std::vector<std::string>{
        "ORCA2_T",   //
//        "ORCA2_U",   //
//        "ORCA2_V",   //
        "eORCA1_T",  //
//        "eORCA025_T",  //
//        "eORCA12_T",  //
    };
    for ( auto gridname : gridnames ) {
        for ( int64_t halo =0; halo < 1; ++halo ) {
            SECTION( gridname + "_halo" + std::to_string(halo) ) {
                auto grid = Grid(gridname);
                auto meshgen_config = grid.meshgenerator() | option::halo(halo);
                atlas::MeshGenerator meshgen(meshgen_config);
                auto partitioner_config = grid.partitioner();
                partitioner_config.set("type", "serial");
                auto partitioner = grid::Partitioner(partitioner_config);
                auto mesh = meshgen.generate(grid, partitioner);
                test::build_remote_index ( mesh );
                //// Bypass for "BuildParallelFields"
                //mesh.nodes().metadata().set( "parallel", true );
                //// Bypass for "BuildPeriodicBoundaries"
                //mesh.metadata().set( "periodic", true );
                REQUIRE( mesh.grid() );
                EXPECT( mesh.grid().name() == gridname );
                idx_t count{0};

                const auto remote_idxs = array::make_indexview<idx_t, 1>(
                                          mesh.nodes().remote_index());
                functionspace::NodeColumns fs{mesh};
                Field field   = fs.createField<double>( option::name( "unswapped ghosts" ) );
                Field field2   = fs.createField<double>( option::name( "remotes < 0" ) );
                auto f        = array::make_view<double, 1>( field );
                auto f2        = array::make_view<double, 1>( field2 );
                const auto ghosts = atlas::array::make_view<int32_t, 1>(
                                      mesh.nodes().ghost());
                const auto lonlat = array::make_view<double, 2>( mesh.nodes().lonlat() );
                for ( idx_t jnode = 0; jnode < mesh.nodes().size(); ++jnode ) {
                    if (ghosts(jnode)) {
                        f( jnode ) = 0;
                    } else {
                        const double lon = lonlat(jnode, 0);
                        const double lat = lonlat(jnode, 1);
                        f( jnode ) = util::function::vortex_rollup(lon, lat, 0.0);
                    }
                }

                fs.haloExchange(field);

                const auto xy = array::make_view<double, 2>( mesh.nodes().xy() );
                const auto glb_idxs = array::make_indexview<int64_t, 1>(mesh.nodes().global_index());
                const auto partition = array::make_view<int32_t, 1>( mesh.nodes().partition() );
                const auto halos = array::make_view<int32_t, 1>( mesh.nodes().halo() );

                const auto master_glb_idxs = array::make_view<gidx_t, 1>( mesh.nodes().field( "master_global_index" ) );
                const auto ij = array::make_view<idx_t, 2>( mesh.nodes().field( "ij" ) );

                for ( idx_t jnode = 0; jnode < mesh.nodes().size(); ++jnode ) {
                    f2(jnode) = 0;
                    const double lon = lonlat(jnode, 0);
                    const double lat = lonlat(jnode, 1);
                    //if (f( jnode )) {
                    if( f(jnode) != util::function::vortex_rollup(lon, lat, 0.0)) {
                      ++count;
                    }
                    if (master_glb_idxs(jnode) == atlas::orca::meshgenerator::TEST_MASTER_GLOBAL_IDX) {
                      std::stringstream strm;
                      strm << "[" << mpi::rank() << "] " << jnode << " : ghost " << ghosts(jnode)
                           << "\n\t i " << ij(jnode, 0) << " j " << ij(jnode, 1)
                           << "\n\t x " << xy(jnode, 0) << " y " << xy(jnode, 1)
                           << "\n\t lon " << lonlat(jnode, 0) << " lat " << lonlat(jnode, 1)
                           << "\n\t remote indices " << remote_idxs(jnode)
                           << "\n\t global indices " << glb_idxs(jnode)
                           << "\n\t master global indices " << master_glb_idxs(jnode)
                           << "\n\t partition " << partition(jnode)
                           << "\n\t halos " << halos(jnode);
                      if (partition(jnode) == mpi::rank()) {
                          strm << " master is ghost? " << ghosts(remote_idxs(jnode));
                      }
                      strm << std::endl;
                      std::cout << strm.str();
                    }
                    if (remote_idxs(jnode) < 0) {
                      std::cout << "[" << mpi::rank() << "] remote_idx < 0 " << jnode
                                << " : ghost " << ghosts(jnode) << std::endl;
                      f2(jnode) = 1;
                    }
                }
                if ( count != 0 ) {
                    Log::info() << "To diagnose problem, uncomment mesh writing here: " << Here() << std::endl;
                    output::Gmsh gmsh(std::string("haloExchange_")+gridname+"_"+std::to_string(halo)+".msh",
                                      Config("coordinates","ij")|Config("info",true));
                    gmsh.write(mesh);
                    gmsh.write(field);
                    gmsh.write(field2);
                }
                EXPECT_EQ( count, 0 );
            }
        }
    }
}


using Unique2Node = std::map<gidx_t, idx_t>;
void build_remote_index(Mesh& mesh) {
    ATLAS_TRACE();

    mesh::Nodes& nodes = mesh.nodes();

    bool parallel = false;
    bool periodic = false;
    nodes.metadata().get("parallel", parallel);
    mesh.metadata().get("periodic", periodic);
    if (parallel | periodic) return;

    auto mpi_size = mpi::size();
    auto mypart   = mpi::rank();
    int nb_nodes = nodes.size();

    // get the indices and partition data
    auto master_glb_idx = array::make_view<gidx_t, 1>(nodes.field("master_global_index"));
    auto glb_idx        = array::make_view<gidx_t, 1>( nodes.global_index() );
    auto ridx    = array::make_indexview<idx_t, 1>( nodes.remote_index() );
    auto part    = array::make_view<int, 1>( nodes.partition() );
    auto ghost   = array::make_view<int, 1>( nodes.ghost() );

    // find the nodes I want to request the data for
    std::vector<std::vector<gidx_t>> send_gidx( mpi_size );
    std::vector<std::vector<int>> req_lidx( mpi_size );

    Unique2Node global2local;
    for ( idx_t jnode = 0; jnode < nodes.size(); ++jnode ) {
        gidx_t uid     = master_glb_idx(jnode);
        if ( (part (jnode) != mypart)
             || ((master_glb_idx( jnode ) != glb_idx( jnode )) &&
                 (part( jnode ) == mypart))
           ) {
            send_gidx[part(jnode)].push_back(uid);
            req_lidx[part(jnode)].push_back(jnode);
            ridx(jnode) = -1;
        } else {
            ridx(jnode) = jnode;
        }
        if (not ghost(jnode)) {
            bool inserted = global2local.insert(std::make_pair(uid, jnode)).second;
            ATLAS_ASSERT(inserted, std::string("index already inserted ") + std::to_string(uid) + ", "
                + std::to_string(jnode) + " at jnode " + std::to_string(global2local[uid]));
        }
    }

    std::vector<std::vector<gidx_t>> recv_gidx( mpi_size );

    // Request data from those indices
    mpi::comm().allToAll( send_gidx, recv_gidx );

    // Find and populate send vector with indices to send
    std::vector<std::vector<int>> send_ridx( mpi_size );
    for ( idx_t p = 0; p < mpi_size; ++p) {
      for ( idx_t i = 0; i < recv_gidx[p].size(); ++i ) {
          idx_t found_idx = -1;
          gidx_t uid     = recv_gidx[p][i];
          Unique2Node::const_iterator found = global2local.find(uid);
          if (found != global2local.end()) {
              found_idx = found->second;
          }
          ATLAS_ASSERT(found_idx != -1,
              "global index not found: " + std::to_string(recv_gidx[p][i]));
          //ATLAS_DEBUG("global index found with remote index: " << ridx(found_idx)
          //    << " partition " << part(found_idx));
          send_ridx[p].push_back(ridx(found_idx));
      }
    }

    std::vector<std::vector<int>> recv_ridx( mpi_size );

    mpi::comm().allToAll( send_ridx, recv_ridx );

    // Fill out missing remote indices
    for ( idx_t p = 0; p < mpi_size; ++p) {
      for ( idx_t i = 0; i < recv_ridx[p].size(); ++i ) {
        ridx(req_lidx[p][i]) = recv_ridx[p][i];
      }
    }

    // sanity check
    for (idx_t jnode = 0; jnode < nb_nodes; ++jnode)
        ATLAS_ASSERT(ridx(jnode) >= 0,
            "ridx not filled with part " + std::to_string(part(jnode)) + " at "
            + std::to_string(jnode));

    mesh.metadata().set( "periodic", true );
    nodes.metadata().set( "parallel", true );
}

}  // namespace test
}  // namespace atlas

int main( int argc, char** argv ) {
    return atlas::test::run( argc, argv );
}
