/*
 * (C) Copyright 2021- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "OrcaMeshGenerator.h"

#include <algorithm>
#include <numeric>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <fstream>
#include <iomanip>

#include "eckit/utils/Hash.h"

#include "atlas/array/Array.h"
#include "atlas/array/ArrayView.h"
#include "atlas/array/IndexView.h"
#include "atlas/array/MakeView.h"
#include "atlas/field/Field.h"
#include "atlas/grid/Distribution.h"
#include "atlas/grid/Partitioner.h"
#include "atlas/grid/Spacing.h"
#include "atlas/grid/StructuredGrid.h"
#include "atlas/library/config.h"
#include "atlas/mesh/ElementType.h"
#include "atlas/mesh/Elements.h"
#include "atlas/mesh/HybridElements.h"
#include "atlas/mesh/Mesh.h"
#include "atlas/mesh/Nodes.h"
#include "atlas/meshgenerator/detail/MeshGeneratorFactory.h"
#include "atlas/meshgenerator/detail/StructuredMeshGenerator.h"
#include "atlas/parallel/mpi/mpi.h"
#include "atlas/runtime/Exception.h"
#include "atlas/runtime/Log.h"
#include "atlas/util/Constants.h"
#include "atlas/util/CoordinateEnums.h"
#include "atlas/util/Geometry.h"
#include "atlas/util/NormaliseLongitude.h"
#include "atlas/util/Topology.h"

#include "atlas-orca/meshgenerator/SurroundingRectangle.h"
#include "atlas-orca/meshgenerator/LocalOrcaGrid.h"


namespace atlas::orca::meshgenerator {

// ORCA2 interesting indices
static const std::vector<std::pair<int, int>> glb_ij_pairs{
    {0, 148},
    {1, 148},
    {0, 147},
    {89, 148},
    {90, 148},
    {91, 148},
    {92, 148},
    {93, 148},
    //89-90, 147
    {89, 147},
    {90, 147},
    //92-101, 147
    {92, 147},
    {93, 147},
    {94, 147},
    {95, 147},
    {96, 147},
    {97, 147},
    {98, 147},
    {99, 147},
    {100, 147},
    {101, 147},
    //89-90, 146
    {89, 146},
    {90, 146},
    //180-181, 148
    {180, 148},
    {181, 148},
    {181, 146},
};

namespace {

StructuredGrid equivalent_regular_grid( const OrcaGrid& orca ) {
    ATLAS_ASSERT( orca );
    // Mimic hole in South pole, and numbering from South to North. patch determines if endpoint is at North Pole
    StructuredGrid::YSpace yspace{grid::LinearSpacing{{-80., 90.}, orca.ny(), true}};  //not patch.at( orca.name() )}};
    // Periodic xspace
    StructuredGrid::XSpace xspace{grid::LinearSpacing{{0., 360.}, orca.nx(), false}};

    return StructuredGrid{xspace, yspace};
}
}  // namespace

struct Nodes {
    array::ArrayView<idx_t, 2> ij;
    array::ArrayView<double, 2> xy;
    array::ArrayView<double, 2> lonlat;
    array::ArrayView<gidx_t, 1> glb_idx;
    array::IndexView<idx_t, 1> remote_idx;
    array::ArrayView<int, 1> part;
    array::ArrayView<int, 1> ghost;
    array::ArrayView<int, 1> halo;
    array::ArrayView<int, 1> node_flags;
    array::ArrayView<int, 1> water;
    array::ArrayView<gidx_t, 1> master_glb_idx;

    util::detail::BitflagsView<int> flags( idx_t i ) { return util::Topology::view( node_flags( i ) ); }

    explicit Nodes( Mesh& mesh ) :
        ij{ array::make_view<idx_t, 2>( mesh.nodes().add(
            Field( "ij", array::make_datatype<idx_t>(), array::make_shape( mesh.nodes().size(), 2 ) ) ) ) },
        xy{ array::make_view<double, 2>( mesh.nodes().xy() ) },
        lonlat{ array::make_view<double, 2>( mesh.nodes().lonlat() ) },
        glb_idx{ array::make_view<gidx_t, 1>( mesh.nodes().global_index() ) },
        remote_idx{ array::make_indexview<idx_t, 1>( mesh.nodes().remote_index() ) },
        part{ array::make_view<int, 1>( mesh.nodes().partition() ) },
        ghost{ array::make_view<int, 1>( mesh.nodes().ghost() ) },
        halo{ array::make_view<int, 1>( mesh.nodes().halo() ) },
        node_flags{ array::make_view<int, 1>( mesh.nodes().flags() ) },
        water{ array::make_view<int, 1>( mesh.nodes().add(
            Field( "water", array::make_datatype<int>(), array::make_shape( mesh.nodes().size() ) ) ) ) },
        master_glb_idx{ array::make_view<gidx_t, 1>( mesh.nodes().add( Field(
            "master_global_index", array::make_datatype<gidx_t>(), array::make_shape( mesh.nodes().size() ) ) ) ) } {}
};

struct Cells {
    array::ArrayView<int, 1> part;
    array::ArrayView<int, 1> halo;
    array::ArrayView<gidx_t, 1> glb_idx;
    array::ArrayView<int, 1> flags_view;
    mesh::HybridElements::Connectivity& node_connectivity;
    util::detail::BitflagsView<int> flags( idx_t i ) { return util::Topology::view( flags_view( i ) ); }
    explicit Cells( Mesh& mesh ) :
        part{ array::make_view<int, 1>( mesh.cells().partition() ) },
        halo{ array::make_view<int, 1>( mesh.cells().halo() ) },
        glb_idx{ array::make_view<gidx_t, 1>( mesh.cells().global_index() ) },
        flags_view{ array::make_view<int, 1>( mesh.cells().flags() ) },
        node_connectivity( mesh.cells().node_connectivity() ) {}
};

void OrcaMeshGenerator::generate( const Grid& grid, const grid::Distribution& distribution, Mesh& mesh ) const {
    ATLAS_TRACE( "OrcaMeshGenerator::generate" );
    using Topology = util::Topology;

    OrcaGrid orca_grid{grid};
    ATLAS_ASSERT( orca_grid );
    ATLAS_ASSERT( !mesh.generated() );

    // global (all processor) configuration information about ORCA grid for the ORCA mesh under construction
    SurroundingRectangle::Configuration SR_cfg;
    SR_cfg.mypart = mypart_;
    SR_cfg.nparts = nparts_;
    SR_cfg.halosize = halosize_;
    SR_cfg.nx_glb = orca_grid.nx();
    SR_cfg.ny_glb = orca_grid.ny();

    SurroundingRectangle SR(distribution, SR_cfg);
    LocalOrcaGrid local_orca(orca_grid, SR);

    // global orca grid dimensions and index limits
    auto ny_orca_halo = orca_grid.ny() + orca_grid.haloNorth() + orca_grid.haloSouth();
    auto nx_orca_halo = orca_grid.nx() + orca_grid.haloEast() + orca_grid.haloWest();
    auto iy_glb_min = -orca_grid.haloSouth();
    auto iy_glb_max = iy_glb_min + ny_orca_halo;
    auto ix_glb_max = orca_grid.haloEast() + orca_grid.nx();
    auto ix_glb_min = -orca_grid.haloWest();

    // clone some grid properties
    setGrid( mesh, grid, distribution );

    // global index of the orca grid
    idx_t glbarray_offset  = -( nx_orca_halo * iy_glb_min ) - ix_glb_min;
    idx_t glbarray_jstride = nx_orca_halo;
    auto orca_global_index = [&]( idx_t i, idx_t j ) {
        ATLAS_ASSERT( i <= ix_glb_max );
        ATLAS_ASSERT( j <= iy_glb_max );
        return glbarray_offset + j * glbarray_jstride + i;
    };

    const bool serial_distribution = (SR_cfg.nparts == 1 || distribution.type() == "serial");

    //---------------------------------------------------

    if ( serial_distribution ) {
        ATLAS_ASSERT_MSG(orca_grid.nx() * orca_grid.ny() == local_orca.nb_real_nodes(),
          std::string("Size of the surrounding rectangle coord system does not match up with the size of the internal space in the orca_grid: ")
          + std::to_string(orca_grid.nx() * orca_grid.nx()) + " != " + std::to_string(local_orca.nb_real_nodes()) );
        ATLAS_ASSERT_MSG(nx_orca_halo == local_orca.nx_orca(),
          std::string("Size of the surrounding rectangle x-space doesn't match up with orca-grid x-space: ")
          + std::to_string(nx_orca_halo) + " != " + std::to_string(local_orca.nx_orca()) );
        ATLAS_ASSERT_MSG(ny_orca_halo == local_orca.ny_orca(),
          std::string("Size of the surrounding rectangle y-space doesn't match up with orca-grid y-space: ")
          + std::to_string(ny_orca_halo) + " != " + std::to_string(local_orca.ny_orca()) );
    }

    // define nodes and associated properties
    mesh.nodes().resize(nx_orca_halo * ny_orca_halo);
    Nodes nodes( mesh );

    // define cells and associated properties
#if ATLAS_TEMPORARY_ELEMENTTYPES
    // DEPRECATED
    mesh.cells().add( new mesh::temporary::Quadrilateral(), SR.nb_cells() );
#else
    // Use this since atlas 0.35.0
    mesh.cells().add( mesh::ElementType::create("Quadrilateral"), SR.nb_cells() );
#endif

    Cells cells( mesh );

    int inode_nonghost = 0;
    int inode_ghost    = 0;


    int ix_pivot = SR_cfg.nx_glb / 2;
    bool patch   = not orca_grid.ghost( ix_pivot + 1, SR_cfg.ny_glb - 1 );

    std::vector<idx_t> node_index( SR.nx()*SR.ny(), -1 );

    std::vector<idx_t> grid_fold_inodes;
    {
        ATLAS_TRACE( "nodes" );

        // loop over nodes and set properties
        inode_nonghost = 0;
        inode_ghost    = SR.nb_real_nodes();  // orca ghost nodes start counting after nonghost nodes

        ATLAS_TRACE_SCOPE( "indexing" )
        for ( idx_t iy = 0; iy < SR.ny(); iy++ ) {
            idx_t iy_glb = SR.iy_min() + iy;
            ATLAS_ASSERT( iy_glb < ny_orca_halo );
            for ( idx_t ix = 0; ix < SR.nx(); ix++ ) {
                idx_t ii = SR.index( ix, iy );
                // node properties
                if ( SR.is_node[ii] ) {
                    // set node counter
                    if ( SR.is_ghost[ii] != 0 ) {
                        node_index[ii] = inode_ghost++;
                        ATLAS_ASSERT_MSG( node_index[ii] < SR.nb_real_nodes(),
                            std::string("node_index[") + std::to_string(ii) + std::string("] ") + std::to_string(node_index[ii]) + " >= " + std::to_string(SR.nb_real_nodes()));
                    }
                    else {
                        node_index[ii] = inode_nonghost++;
                        ATLAS_ASSERT( node_index[ii] < SR.nb_real_nodes() );
                    }
                }
            }
        }


        ATLAS_TRACE_SCOPE( "filling" )
        //atlas_omp_parallel_for( idx_t iy = 0; iy < SR.ny(); iy++ ) {
        for( idx_t iy = 0; iy < SR.ny(); iy++ ) {
            idx_t iy_glb = SR.iy_min() + iy;
            ATLAS_ASSERT( iy_glb < ny_orca_halo );
            double lon00 = orca_grid.xy( 0, 0 ).x();
            double west  = lon00 - 90.;

            auto normalise_lon00 = util::NormaliseLongitude( lon00 - 180. );
            double lon1          = normalise_lon00( orca_grid.xy( 1, iy_glb ).x() );
            if ( lon1 < lon00 - 10. ) {
                west = lon00 - 20.;
            }

            auto normalise_lon_first_half  = util::NormaliseLongitude{west};
            auto normalise_lon_second_half = util::NormaliseLongitude{lon00 + 90.};
            for ( idx_t ix = 0; ix < SR.nx(); ix++ ) {
                idx_t ix_glb   = SR.ix_min() + ix;
                idx_t ix_glb_master = ix_glb;
                auto normalise = [&]( double _xy[2] ) {
                    if ( ix_glb_master < SR_cfg.nx_glb / 2 ) {
                        _xy[LON] = normalise_lon_first_half( _xy[LON] );
                    }
                    else {
                        _xy[LON] = normalise_lon_second_half( _xy[LON] );
                    }
                };

                idx_t ii = SR.index( ix, iy );
                // node properties
                if ( SR.is_node[ii] ) {
                    idx_t inode = node_index[ii];

                    // ghost nodes
                    nodes.ghost( inode ) = SR.is_ghost[ii];
                    if ( iy_glb > 0 or ix_glb < 0 ) {
                        nodes.ghost( inode ) = nodes.ghost( inode ) || orca_grid.ghost( ix_glb, iy_glb );
                    }

                    // flags
                    auto flags = nodes.flags( inode );
                    flags.reset();

                    // global index
                    nodes.glb_idx( inode ) = orca_global_index( ix_glb, iy_glb ) + 1;  // no periodic point

                    // grid ij coordinates
                    nodes.ij( inode, XX ) = ix_glb;
                    nodes.ij( inode, YY ) = iy_glb;

                    double _xy[2];

                    // grid xy coordinates 
                    orca_grid.xy( ix_glb, iy_glb, _xy );

                    nodes.xy( inode, LON ) = _xy[LON];
                    nodes.xy( inode, LAT ) = _xy[LAT];

                    // geographic coordinates (normalised)
                    normalise( _xy );
                    nodes.lonlat( inode, LON ) = _xy[LON];
                    nodes.lonlat( inode, LAT ) = _xy[LAT];

                    // part and remote_idx
                    nodes.part( inode )           = SR.parts[ii];
                    nodes.remote_idx( inode )     = inode;
                    nodes.master_glb_idx( inode ) = nodes.glb_idx( inode );
                    if ( nodes.ghost( inode ) ) {
                        gidx_t master_idx             = orca_grid.periodicIndex( ix_glb, iy_glb );
                        nodes.master_glb_idx( inode ) = master_idx + 1;
                        idx_t master_i, master_j;
                        orca_grid.index2ij( master_idx, master_i, master_j );
                        nodes.part( inode ) = SR.partition( master_i, master_j );
                        flags.set( Topology::GHOST );
                        nodes.remote_idx( inode ) = serial_distribution ? static_cast<int>( master_idx ) : -1;

                        if( nodes.glb_idx(inode) != nodes.master_glb_idx(inode) ) {
                            if ( ix_glb >= SR_cfg.nx_glb - orca_grid.haloWest() ) {
                                flags.set( Topology::PERIODIC );
                            }
                            else if ( ix_glb < orca_grid.haloEast() - 1 ) {
                                flags.set( Topology::PERIODIC );
                            }
                            if ( iy_glb >= SR_cfg.ny_glb - orca_grid.haloNorth() - 1 ) {
                                flags.set( Topology::PERIODIC );
                                if ( _xy[LON] > lon00 + 90. ) {
                                    flags.set( Topology::EAST );
                                }
                                else {
                                    flags.set( Topology::WEST );
                                }
                            }

                            if ( flags.check( Topology::PERIODIC ) ) {
                                // It can still happen that nodes were flagged as periodic wrongly
                                // e.g. where the grid folds into itself

                                idx_t iy_glb_master = 0;
                                double xy_master[2];
                                orca_grid.index2ij( master_idx, ix_glb_master, iy_glb_master );
                                orca_grid.lonlat(ix_glb_master,iy_glb_master,xy_master);
                                normalise( xy_master );
                                if( std::abs(xy_master[LON] - _xy[LON]) < 1.e-12 ) {
                                    flags.unset(Topology::PERIODIC);
                                    if (( std::abs(xy_master[LAT] - _xy[LAT]) < 1.e-12 ) &&
                                        ( iy_glb >= SR_cfg.ny_glb - orca_grid.haloNorth() - 1 ))
                                        grid_fold_inodes.push_back(inode);
                                }
                            }
                        }
                    }

                    flags.set( orca_grid.land( ix_glb, iy_glb ) ? Topology::LAND : Topology::WATER );

                    if ( ix_glb <= 0 ) {
                        flags.set( Topology::BC | Topology::WEST );
                    }
                    else if ( ix_glb >= SR_cfg.nx_glb ) {
                        flags.set( Topology::BC | Topology::EAST );
                    }

                    nodes.water( inode ) = orca_grid.water( ix_glb, iy_glb );
                    nodes.halo( inode ) = SR.halo[ii];
                }
            }
        }
    }
    std::vector<idx_t> cell_index( SR.nx()*SR.ny() );
    // loop over nodes and define cells
    {
        ATLAS_TRACE( "elements" );
        idx_t jcell = 0;
        ATLAS_TRACE_SCOPE( "indexing" );
        for ( idx_t iy = 0; iy < SR.ny() - 1; iy++ ) {      // don't loop into ghost/periodicity row
            for ( idx_t ix = 0; ix < SR.nx() - 1; ix++ ) {  // don't loop into ghost/periodicity column
                idx_t ii = SR.index( ix, iy );
                if ( SR.is_ghost[ii] == 0 ) {
                    cell_index[ii] = jcell++;
                }
            }
        }

        ATLAS_TRACE_SCOPE( "filling" )
        //atlas_omp_parallel_for( idx_t iy = 0; iy < SR.ny() - 1; iy++ ) {  // don't loop into ghost/periodicity row
        for( idx_t iy = 0; iy < SR.ny() - 1; iy++ ) {  // don't loop into ghost/periodicity row
            for ( idx_t ix = 0; ix < SR.nx() - 1; ix++ ) {                // don't loop into ghost/periodicity column
                idx_t ii   = SR.index( ix, iy );
                int ix_glb = SR.ix_min() + ix;
                int iy_glb = SR.iy_min() + iy;
                if ( !SR.is_ghost[ii] ) {
                    idx_t jcell = cell_index[ii];

                    // define cell corners (local indices)
                    std::array<idx_t, 4> quad_nodes{};
                    quad_nodes[0] = node_index[SR.index( ix, iy )];          // lower left
                    quad_nodes[1] = node_index[SR.index( ix + 1, iy )];      // lower right
                    quad_nodes[2] = node_index[SR.index( ix + 1, iy + 1 )];  // upper right
                    quad_nodes[3] = node_index[SR.index( ix, iy + 1 )];      // upper left

                    cells.flags( jcell ).reset();

                    cells.node_connectivity.set( jcell, quad_nodes.data() );
                    cells.part( jcell )    = nodes.part( quad_nodes[0] );
                    cells.glb_idx( jcell ) = ( iy_glb - iy_glb_min ) * ( nx_orca_halo - 1 ) + ( ix_glb - ix_glb_min ) + 1;

                    if ( iy_glb >= SR_cfg.ny_glb - 1 ) {
                        cells.flags( jcell ).set( Topology::GHOST );
                        if ( patch && ix_glb < ix_pivot ) {                     // case of eg ORCA1_T
                            cells.part( jcell ) = nodes.part( quad_nodes[0] );  // lower left
                            cells.flags( jcell ).unset( Topology::GHOST );
                        }
                        else {                                                  // case of eg ORCA2_T
                            cells.part( jcell ) = nodes.part( quad_nodes[2] );  // upper right
                        }
                    }

                    bool elem_contains_water_point = [&] {
                        for ( idx_t inode : quad_nodes ) {
                            if ( nodes.flags( inode ).check( Topology::WATER ) ) {
                                return true;
                            }
                        }
                        return false;
                    }();
                    bool elem_contains_land_point = [&] {
                        for ( idx_t inode : quad_nodes ) {
                            if ( nodes.flags( inode ).check( Topology::LAND ) ) {
                                return true;
                            }
                        }
                        return false;
                    }();
                    cells.halo( jcell ) = [&] {
                        int h = 0;
                        for ( idx_t inode : quad_nodes ) {
                            h = std::max( h, nodes.halo( inode ) );
                        }
                        if ( iy_glb < 0 ) {
                            h = 0;
                            if ( ix_glb < 0 ) {
                                h = -ix_glb;
                            }
                            else if ( ix_glb >= SR_cfg.nx_glb ) {
                                h = ix_glb - ( SR_cfg.nx_glb - 1 );
                            }
                        }
                        return h;
                    }();

                    if ( elem_contains_water_point ) {
                        cells.flags( jcell ).set( Topology::WATER );
                    }
                    if ( elem_contains_land_point ) {
                        cells.flags( jcell ).set( Topology::LAND );
                    }
                    if ( orca_grid.invalidElement( SR.ix_min() + ix, SR.iy_min() + iy ) ) {
                        cells.flags( jcell ).set( Topology::INVALID );
                    }
                }
            }
        }
    }
    ATLAS_DEBUG_VAR( serial_distribution );
    if ( serial_distribution ) {
        // Bypass for "BuildParallelFields"
        mesh.nodes().metadata().set( "parallel", true );

        // Bypass for "BuildPeriodicBoundaries"
        mesh.metadata().set( "periodic", true );
    }
    else {
        ATLAS_DEBUG( "build_remote_index" );
        build_remote_index( mesh );
    }

    // Degenerate points in the ORCA mesh mean that the standard BuildHalo
    // methods for updating halo sizes will not work.
    mesh.metadata().set("halo_locked", true);
    mesh.metadata().set("halo", halosize_);
    mesh.nodes().metadata().set<size_t>( "NbRealPts", SR.nb_real_nodes() );
    mesh.nodes().metadata().set<size_t>( "NbVirtualPts", SR.nb_ghost_nodes() );
}

using Unique2Node = std::map<gidx_t, idx_t>;
void OrcaMeshGenerator::build_remote_index(Mesh& mesh) const {
    ATLAS_TRACE();

    mesh::Nodes& nodes = mesh.nodes();

    bool parallel = false;
    bool periodic = false;
    nodes.metadata().get( "parallel", parallel );
    mesh.metadata().get( "periodic", periodic );
    if ( parallel || periodic ) {
        ATLAS_DEBUG( "build_remote_index: already parallel, return" );
        return;
    }

    auto mpi_size = mpi::size();
    auto mypart   = mpi::rank();
    int nb_nodes  = nodes.size();

    // get the indices and partition data
    auto master_glb_idx = array::make_view<gidx_t, 1>( nodes.field( "master_global_index" ) );
    auto glb_idx        = array::make_view<gidx_t, 1>( nodes.global_index() );
    auto ridx           = array::make_indexview<idx_t, 1>( nodes.remote_index() );
    auto part           = array::make_view<int, 1>( nodes.partition() );
    auto ghost          = array::make_view<int, 1>( nodes.ghost() );

    // find the nodes I want to request the data for
    std::vector<std::vector<gidx_t>> send_uid( mpi_size );
    std::vector<std::vector<int>> req_lidx( mpi_size );

    Unique2Node global2local;
    for ( idx_t jnode = 0; jnode < nodes.size(); ++jnode ) {
        gidx_t uid = master_glb_idx( jnode );
        if ( ( part( jnode ) != mypart ) ||
             ( ( master_glb_idx( jnode ) != glb_idx( jnode ) ) && ( part( jnode ) == mypart ) ) ) {
            send_uid[part( jnode )].push_back( uid );
            req_lidx[part( jnode )].push_back( jnode );
            ridx( jnode ) = -1;
        }
        else {
            ridx( jnode ) = jnode;
        }
        if ( ghost( jnode ) == 0 ) {
            bool inserted = global2local.insert( std::make_pair( uid, jnode ) ).second;
            ATLAS_ASSERT( inserted, std::string( "index already inserted " ) + std::to_string( uid ) + ", " +
                                        std::to_string( jnode ) + " at jnode " + std::to_string( global2local[uid] ) );
        }
    }

    std::vector<std::vector<gidx_t>> recv_uid( mpi_size );

    // Request data from those indices
    mpi::comm().allToAll( send_uid, recv_uid );

    // Find and populate send vector with indices to send
    std::vector<std::vector<int>> send_ridx( mpi_size );
    std::vector<std::vector<int>> send_gidx( mpi_size );
    std::vector<std::vector<int>> send_part( mpi_size );
    for ( idx_t p = 0; p < mpi_size; ++p ) {
        for ( idx_t i = 0; i < recv_uid[p].size(); ++i ) {
            idx_t found_idx = -1;
            gidx_t uid      = recv_uid[p][i];
            if ( auto found = global2local.find( uid ); found != global2local.end() ) {
                found_idx = found->second;
            }

            ATLAS_ASSERT( found_idx != -1, "master global index not found: " + std::to_string( recv_uid[p][i] ) );
            send_ridx[p].push_back( ridx( found_idx ) );
            send_gidx[p].push_back( static_cast<int>( glb_idx( found_idx ) ) );
            send_part[p].push_back( part( found_idx ) );
        }
    }

    std::vector<std::vector<int>> recv_ridx( mpi_size );
    std::vector<std::vector<int>> recv_gidx( mpi_size );
    std::vector<std::vector<int>> recv_part( mpi_size );

    mpi::comm().allToAll( send_ridx, recv_ridx );
    mpi::comm().allToAll( send_gidx, recv_gidx );
    mpi::comm().allToAll( send_part, recv_part );

    // Fill out missing remote indices
    for ( idx_t p = 0; p < mpi_size; ++p ) {
        for ( idx_t i = 0; i < recv_ridx[p].size(); ++i ) {
            ridx( req_lidx[p][i] )    = recv_ridx[p][i];
            glb_idx( req_lidx[p][i] ) = recv_gidx[p][i];
            part( req_lidx[p][i] )    = recv_part[p][i];
        }
    }

    // sanity check
    for ( idx_t jnode = 0; jnode < nb_nodes; ++jnode ) {
        ATLAS_ASSERT( ridx( jnode ) >= 0, "ridx not filled with part " + std::to_string( part( jnode ) ) + " at " +
                                              std::to_string( jnode ) );
    }

    mesh.metadata().set( "periodic", true );
    nodes.metadata().set( "parallel", true );
}

OrcaMeshGenerator::OrcaMeshGenerator( const eckit::Parametrisation& config ) {
    config.get( "partition", mypart_ = mpi::rank() );
    config.get( "partitions", nparts_ = mpi::size() );
    config.get( "halo", halosize_);
    if (halosize_ < 0 || halosize_ > 1)
      throw_NotImplemented("Only halo sizes 0 or 1 ORCA grids are currently supported", Here());
}

void OrcaMeshGenerator::generate( const Grid& grid, const grid::Partitioner& partitioner, Mesh& mesh ) const {
    std::unordered_set<std::string> valid_distributions = {"serial", "checkerboard", "equal_regions", "equal_area"};
    ATLAS_ASSERT(valid_distributions.find(partitioner.type()) != valid_distributions.end(),
                 partitioner.type() + " is not an implemented distribution type. "
                 + "Valid types are 'serial', 'checkerboard' or 'equal_regions', 'equal_area'");
    if (partitioner.type() == "serial" && halosize_ > 1)
      throw_NotImplemented("halo size must be zero for 'serial' distribution type ORCA grids", Here());
    auto regular_grid = equivalent_regular_grid( grid );
    auto distribution = grid::Distribution( regular_grid, partitioner );
    generate( grid, distribution, mesh );
}

void OrcaMeshGenerator::generate( const Grid& grid, Mesh& mesh ) const {
    generate( grid, grid::Partitioner( grid.partitioner() ), mesh );
}

void OrcaMeshGenerator::hash( eckit::Hash& h ) const {
    h.add( "OrcaMeshGenerator" );
}

namespace {
atlas::meshgenerator::MeshGeneratorBuilder<OrcaMeshGenerator> __OrcaMeshGenerator( "orca" );
}

}  // namespace atlas::orca::meshgenerator
