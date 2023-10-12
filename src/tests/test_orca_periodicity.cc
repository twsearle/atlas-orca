/*
 * (C) Copyright 2021- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "eckit/log/Bytes.h"
#include "eckit/system/ResourceUsage.h"

#include "atlas/functionspace/NodeColumns.h"
#include "atlas/grid.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/output/Gmsh.h"
#include "atlas/util/Config.h"

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

CASE( "test orca periodicity" ) {
    auto gridnames = std::vector<std::string>{
        "ORCA2_T",   //
        "eORCA1_T",  //
        "eORCA025_T",  //
    };
    for ( auto gridname : gridnames ) {
        SECTION( gridname ) {
            int64_t halo = 0;
            auto agrid = Grid(gridname);

            auto meshgen_config = agrid.meshgenerator() | option::halo(halo);
            atlas::MeshGenerator meshgen(meshgen_config);
            auto partitioner_config = agrid.partitioner();
            partitioner_config.set("type", "serial");
            auto partitioner = grid::Partitioner(partitioner_config);
            auto mesh = meshgen.generate(agrid, partitioner);
            REQUIRE( mesh.grid() );
            EXPECT( mesh.grid().name() == gridname );

            OrcaGrid grid = mesh.grid();
            auto periodicity = grid.get()->periodicity();
            auto pivot = periodicity.pivot();
            auto midpoint = pivot[0];
            auto nx = grid.nx();
            auto ny = grid.ny();
            auto ij = array::make_view<idx_t, 2>( mesh.nodes().field( "ij" ) );
            auto master_glb_idx_field = mesh.nodes().field("master_global_index");

            size_t count{0};
            functionspace::NodeColumns fs{mesh};
            Field field     = fs.createField<double>( option::name( "mismatch periodicity" ) );
            auto f          = array::make_view<double, 1>( field );
            Field ifield    = fs.createField<double>( option::name( "i coord" ) );
            auto icoord     = array::make_view<double, 1>( ifield );
            Field jfield    = fs.createField<double>( option::name( "j coord" ) );
            auto jcoord     = array::make_view<double, 1>( jfield );
            Field ipfield   = fs.createField<double>( option::name( "i periodic" ) );
            auto ipcoord    = array::make_view<double, 1>( ipfield );
            Field jpfield   = fs.createField<double>( option::name( "j periodic" ) );
            auto jpcoord    = array::make_view<double, 1>( jpfield );
            Field ipdfield  = fs.createField<double>( option::name( "i periodic diff" ) );
            auto ipdcoord   = array::make_view<double, 1>( ipdfield );
            Field jpdfield  = fs.createField<double>( option::name( "j periodic diff" ) );
            auto jpdcoord   = array::make_view<double, 1>( jpdfield );
            Field idxdfield = fs.createField<double>( option::name( "idx remote_idx diff" ) );
            auto idxdcoord  = array::make_view<double, 1>( idxdfield );
            Field ghostrecurrentfield = fs.createField<double>( option::name( "ghost points to halo" ) );
            auto f_ghost_recurrent = array::make_view<double, 1>( ghostrecurrentfield );
            auto remote_idx = array::make_indexview<idx_t, 1>( mesh.nodes().remote_index() );
            auto parts      = atlas::array::make_view<int32_t, 1>(
                                mesh.nodes().partition());
            auto halos      = atlas::array::make_view<int32_t, 1>(
                                mesh.nodes().halo());
            auto ghosts     = atlas::array::make_view<int32_t, 1>(
                                mesh.nodes().ghost());
            auto mypart   = mpi::rank();
            for ( idx_t jnode = 0; jnode < mesh.nodes().size(); ++jnode ) {
                auto ijLoc = orca::PointIJ{ij( jnode, 0 ), ij( jnode, 1 )};
                icoord(jnode) = ijLoc.i;
                jcoord(jnode) = ijLoc.j;
                auto p = periodicity(ijLoc);
                ipcoord(jnode) = p.i;
                jpcoord(jnode) = p.j;
                ipdcoord(jnode) = ijLoc.i - p.i;
                jpdcoord(jnode) = ijLoc.j - p.j;
                idxdcoord(jnode) = jnode - remote_idx(jnode);
                f_ghost_recurrent( jnode ) = 0;
                if (ghosts(jnode) &&
                    (parts(jnode) == mypart) &&
                    halos(remote_idx(jnode))) {
                    f_ghost_recurrent( jnode ) = 1;
                }
                f(jnode) = 0;
                if (ijLoc.i < 0) {
                  // this should be west boundary located ghost-periodic points?
                  if (p.i != nx) {
                    ++count;
                    f(jnode) = 1;
                  }
                } else if (ijLoc.j == ny) {
                  // this should be northfold located ghost-periodic points on the top row
                  if (ijLoc.i < midpoint) {
                    if ((p.i != nx - ijLoc.i) || (p.j != ny - 2)) {
                      ++count;
                      f(jnode) = 2;
                    }
                  } else {
                    if ((nx - p.i != ijLoc.i) || (p.j != ny - 2)) {
                      ++count;
                      f(jnode) = 2;
                    }
                  }
                } else if (ijLoc.j == ny - 1) {
                  // this should be northfold located ghost-periodic points on
                  // the second from top row
                  if (ijLoc.i < midpoint) {
                    if ((p.i != nx - ijLoc.i) || (p.j != ny - 1)) {
                      ++count;
                      f(jnode) = 2;
                    }
                  } else {
                    if ((nx - p.i != ijLoc.i) || (p.j != ny - 1)) {
                      ++count;
                      f(jnode) = 2;
                    }
                  }
                } else {
                    if (p.i != ijLoc.i && p.j != ijLoc.j) {
                      ++count;
                      f(jnode) = 3;
                    }
                }
            }
            if ( count != 0 ) {
                Log::info() << "To diagnose problem, uncomment mesh writing here: " << Here() << std::endl;
                output::Gmsh gmsh(gridname+".msh",Config("coordinates","ij")|Config("ghost",true)|Config("info",true));
                gmsh.write(mesh);
                gmsh.write(field);
                gmsh.write(master_glb_idx_field);
                gmsh.write(ifield);
                gmsh.write(jfield);
                gmsh.write(ipfield);
                gmsh.write(jpfield);
                gmsh.write(ipdfield);
                gmsh.write(jpdfield);
                gmsh.write(idxdfield);
                gmsh.write(ghostrecurrentfield);
            }
            EXPECT_EQ( count, 0 );
        }
    }
}


}  // namespace test
}  // namespace atlas

int main( int argc, char** argv ) {
    return atlas::test::run( argc, argv );
}
