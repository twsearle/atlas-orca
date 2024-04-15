/*
 * (C) Copyright 2021- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <limits>

#include "atlas/util/Topology.h"
#include "atlas-orca/meshgenerator/LocalOrcaGrid.h"


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace eckit {
class Parametrisation;
}
#endif

namespace atlas::orca::meshgenerator {

//----------------------------------------------------------------------------------------------------------------------
LocalOrcaGrid::LocalOrcaGrid(const OrcaGrid& grid, const SurroundingRectangle& rectangle) :
        orca_(grid) {

  ix_orca_min_ = rectangle.ix_min();
  ix_orca_max_ = rectangle.ix_max();
  iy_orca_min_ = rectangle.iy_min();
  iy_orca_max_ = rectangle.iy_max();

  std::cout << " rectangle.nx() " << rectangle.nx()
            << " rectangle.ny() " << rectangle.ny()
            << " rectangle.ix_min() " << rectangle.ix_min()
            << " rectangle.ix_max() " << rectangle.ix_max()
            << " rectangle.iy_min() " << rectangle.iy_min()
            << " rectangle.iy_max() " << rectangle.iy_max() << std::endl;

  // Add on the orca halo points if we are at the edge of the orca grid.
  if (rectangle.ix_min() <= 0) {
    ix_orca_min_ -= orca_.haloWest();
  }
  if (rectangle.ix_max() >= orca_.nx()-1) {
    ix_orca_max_ += orca_.haloEast();
  }
  if (rectangle.iy_min() <= 0) {
    iy_orca_min_ -= orca_.haloSouth();
  }
  if (rectangle.iy_max() >= orca_.ny()-1) {
    iy_orca_max_ += orca_.haloNorth();
  }

  // TODO: remove this, only here to clamp the maximum within orca grid for
  // halo=0 replication
  {
    int ix_glb_max = orca_.nx() + orca_.haloEast();
    int iy_glb_max = orca_.ny() + orca_.haloNorth();
    ix_orca_max_ = std::min( ix_glb_max, ix_orca_max_ );
    iy_orca_max_ = std::min( iy_glb_max, iy_orca_max_ );
  }

  std::cout << " orca_.nx() " << orca_.nx()
            << " orca_.ny() " << orca_.ny() << std::endl;

  std::cout << " ix_orca_min_ " <<  ix_orca_min_
            << " ix_orca_max_ " <<  ix_orca_max_
            << " iy_orca_min_ " <<  iy_orca_min_
            << " iy_orca_max_ " <<  iy_orca_max_ << std::endl;

  // dimensions of the rectangle including the ORCA halo points
  // NOTE: +1 because the size of the dimension is one bigger than index of the last element
  nx_orca_ = ix_orca_max_ - ix_orca_min_ + 1;
  ny_orca_ = iy_orca_max_ - iy_orca_min_ + 1;
  size_ = nx_orca_ * ny_orca_;
  std::cout << "size_ " << size_ << std::endl;

  // partitions and local indices in surrounding rectangle
  parts.resize( size_, -1 );
  halo.resize( size_, 0 );
  is_node.resize( size_, false );
  is_ghost.resize( size_, true );
  nb_real_nodes_ = 0;
  nb_ghost_nodes_ = 0;
  uint16_t nb_halo_nodes = 0;
  {
    //atlas_omp_parallel_for( idx_t iy = 0; iy < ny_; iy++ )
    for( size_t iy = 0; iy < ny_orca_; iy++ ) {
      for ( size_t ix = 0; ix < nx_orca_; ix++ ) {
        idx_t ii = index( ix, iy );
        idx_t reg_ii = 0;
        // clamp information to values in the surrounding rectangle if they lie in the orca halo
        // TODO: Use orca grid periodicity information to inform ghost/halo/partition info?
        reg_ii = rectangle.index(ix < rectangle.nx() ? ix : rectangle.nx()-1,
                                 iy < rectangle.ny() ? iy : rectangle.ny()-1);
        parts.at( ii )    = rectangle.parts.at( reg_ii );
        halo.at( ii )     = rectangle.halo.at( reg_ii );
        is_ghost.at( ii ) = rectangle.is_ghost.at( reg_ii );
        const auto ij_glb = this->global_ij( ix, iy );
        if ( ij_glb.j > 0 or ij_glb.i < 0 ) {
          is_ghost.at( ii ) = is_ghost.at( ii ) || orca_.ghost( ij_glb.i, ij_glb.j );
        }
        if ( is_ghost.at( ii ) ) {
          ++nb_ghost_nodes_;
          if ( halo[ii] != 0)
            ++nb_halo_nodes;
        } else {
          ++nb_real_nodes_;
        }
      }
    }
  }

  // determine number of cells and number of nodes
  {
    std::vector<int> is_cell(size_, false);
    auto mark_node_used = [&]( int ix, int iy ) {
      idx_t ii = index( ix, iy );
      if ( !is_node.at(ii) ) {
        ++nb_used_nodes_;
        is_node.at(ii) = true;
      }
    };
    auto mark_cell_used = [&]( int ix, int iy ) {
      idx_t ii = index( ix, iy );
      if ( !is_cell.at(ii) ) {
        ++nb_cells_;
        is_cell.at(ii) = true;
      }
    };
    // Loop over all elements to determine which are required
    nb_cells_ = 0;
    nb_used_nodes_ = 0;
    for ( idx_t iy = 0; iy < ny_orca_-1; iy++ ) {
      for ( idx_t ix = 0; ix < nx_orca_-1; ix++ ) {
        if ( ! is_ghost[index( ix, iy )]) {
          mark_cell_used( ix, iy );
          mark_node_used( ix, iy );
          mark_node_used( ix + 1, iy );
          mark_node_used( ix + 1, iy + 1 );
          mark_node_used( ix, iy + 1 );
        } // if not ghost index
      }
    }
  }

  // setup normalisation objects
  {
    lon00_ = orca_.xy( 0, 0 ).x();
    lon00_normaliser_ = util::NormaliseLongitude(lon00_ - 180. );
  }
}
int LocalOrcaGrid::index( idx_t ix, idx_t iy ) const {
  ATLAS_ASSERT_MSG(static_cast<size_t>(ix) < nx_orca_,
     std::string("ix >= nx_orca_: ") + std::to_string(ix) + " >= " + std::to_string(nx_orca_));
  ATLAS_ASSERT_MSG(static_cast<size_t>(iy) < ny_orca_,
    std::string("iy >= ny_orca_: ") + std::to_string(iy) + " >= " + std::to_string(ny_orca_));
  return iy * nx_orca_ + ix;
}

PointIJ LocalOrcaGrid::global_ij( idx_t ix, idx_t iy ) const {
  return PointIJ(ix_orca_min_ + ix, iy_orca_min_ + iy);
}

const PointXY& LocalOrcaGrid::grid_xy( idx_t ix, idx_t iy ) const {
  const auto ij = this->global_ij( ix, iy );
  return orca_.xy( ij.i, ij.j );
}

PointXY LocalOrcaGrid::normalised_grid_xy( idx_t ix, idx_t iy ) const {
  double west  = lon00_ - 90.;
  const auto ij = this->global_ij( ix, iy );
  const PointXY xy = orca_.xy( ij.i, ij.j );
  double lon1 = lon00_normaliser_( orca_.xy( 1, ij.j ).x() );
  if ( lon1 < lon00_ - 10. ) {
      west = lon00_ - 20.;
  }

  if ( ij.i < nx_orca_ / 2 ) {
    auto lon_first_half_normaliser  = util::NormaliseLongitude{west};
    return PointXY( lon_first_half_normaliser( xy.x() ), xy.y() );
  } else {
    auto lon_second_half_normaliser = util::NormaliseLongitude{lon00_ + 90.};
    return PointXY( lon_second_half_normaliser( xy.x() ), xy.y() );
  }
}

// unique global index of the orca grid excluding orca grid halos (orca halo points are wrapped to their master index).
// Note: right now need to add +1 to this to fill the field with corresponding name.
gidx_t LocalOrcaGrid::master_global_index( idx_t ix, idx_t iy ) const {
  auto ij = this->global_ij(ix, iy);
  return orca_.periodicIndex( ij.i, ij.j );
}

PointIJ LocalOrcaGrid::master_global_ij( idx_t ix, idx_t iy ) const {
  const auto master_idx = this->master_global_index( ix, iy );
  idx_t ix_glb_master, iy_glb_master;
  orca_.index2ij( master_idx, ix_glb_master, iy_glb_master );
  return PointIJ(ix_glb_master, iy_glb_master);
}

PointLonLat LocalOrcaGrid::normalised_grid_master_lonlat( idx_t ix, idx_t iy ) const {
  double west  = lon00_ - 90.;
  const auto ij = this->global_ij( ix, iy );
  const auto master_ij = this->master_global_ij( ix, iy );

  const PointLonLat lonlat = orca_.lonlat( master_ij.i, master_ij.j );

  double lon1 = lon00_normaliser_( orca_.xy( 1, ij.j ).x() );
  if ( lon1 < lon00_ - 10. ) {
      west = lon00_ - 20.;
  }

  if ( ij.i < nx_orca_ / 2 ) {
    auto lon_first_half_normaliser  = util::NormaliseLongitude{west};
    return PointLonLat( lon_first_half_normaliser( lonlat.lon() ), lonlat.lat() );
  } else {
    auto lon_second_half_normaliser = util::NormaliseLongitude{lon00_ + 90.};
    return PointLonLat( lon_second_half_normaliser( lonlat.lon() ), lonlat.lat() );
  }
}

// unique global index of the orca grid including orca grid halos.  This needs
// to wrap points back into the orca grid, but is subtly different from
// OrcaGrid.PeriodicIndex as it will only wrap points that are outside of the
// orca grid halos
idx_t LocalOrcaGrid::orca_haloed_global_grid_index( idx_t ix, idx_t iy ) const {
  // global grid properties
  auto iy_glb_min = -orca_.haloSouth();
  auto ix_glb_min = -orca_.haloWest();
  idx_t glbarray_offset  = -( nx_orca_ * iy_glb_min ) - ix_glb_min;
  idx_t glbarray_jstride = nx_orca_;

  auto ij = this->global_ij( ix, iy );

  // wrap points outside of orca_grid halo back into the orca grid.
  if( (ij.i > ix_glb_min + nx_orca_) || (ij.j > iy_glb_min + ny_orca_) ) {
    gidx_t p_idx = orca_.periodicIndex(ij.i, ij.j);
    idx_t i, j;
    orca_.index2ij(p_idx, i, j);
    ij.i = i;
    ij.j = j;
  }

  ATLAS_ASSERT_MSG( ij.i <= ix_glb_min + nx_orca_,
                    std::to_string(ij.i) + std::string(" > ") + std::to_string(ix_glb_min + nx_orca_) );
  ATLAS_ASSERT_MSG( ij.j <= iy_glb_min + ny_orca_,
                    std::to_string(ij.j) + std::string(" > ") + std::to_string(iy_glb_min + ny_orca_) );
  return glbarray_offset + ij.j * glbarray_jstride + ij.i;
}

void LocalOrcaGrid::flags( idx_t ix, idx_t iy, util::detail::BitflagsView<int>& flag_view ) const {
  flag_view.reset();
  const auto ij_glb = this->global_ij( ix, iy );
  if ( this->is_ghost[this->index(ix, iy)] ) {
    flag_view.set( util::Topology::GHOST );
    if( this->orca_haloed_global_grid_index( ix, iy ) !=
        this->master_global_index( ix, iy ) ) {
      const auto normalised_xy = this->normalised_grid_xy( ix, iy );
      if ( ij_glb.i >= orca_.nx() - orca_.haloWest() ) {
        flag_view.set( util::Topology::PERIODIC );
      }
      else if ( ij_glb.i < orca_.haloEast() - 1 ) {
        flag_view.set( util::Topology::PERIODIC );
      }
      if ( ij_glb.j >= orca_.ny() - orca_.haloNorth() - 1 ) {
          flag_view.set( util::Topology::PERIODIC );
          if ( normalised_xy.x() > lon00_ + 90. ) {
              flag_view.set( util::Topology::EAST );
          }
          else {
              flag_view.set( util::Topology::WEST );
          }
      }

      if ( flag_view.check( util::Topology::PERIODIC ) ) {
        // It can still happen that nodes were flagged as periodic wrongly
        // e.g. where the grid folds into itself
        auto lonlat_master = this->normalised_grid_master_lonlat(ix, iy);
        if( std::abs(lonlat_master.lon() - normalised_xy.x()) < 1.e-12 ) {
            flag_view.unset( util::Topology::PERIODIC );
            // if (( std::abs(lonlat_master.lat() - normalised_xy.y()) < 1.e-12 ) &&
            //     ( ij_glb.j >= orca_.ny() - orca_.haloNorth() - 1 ))
            //     grid_fold_inodes.push_back(inode);
        }
      }
    }
  }

  flag_view.set( orca_.land( ij_glb.i, ij_glb.j ) ? util::Topology::LAND : util::Topology::WATER );

  if ( ij_glb.i <= 0 ) {
    flag_view.set( util::Topology::BC | util::Topology::WEST );
  }
  else if ( ij_glb.j >= orca_.nx() ) {
    flag_view.set( util::Topology::BC | util::Topology::EAST );
  }
}
bool LocalOrcaGrid::water( idx_t ix, idx_t iy ) const {
  const auto ij_glb = this->global_ij( ix, iy );
  return orca_.water( ij_glb.i, ij_glb.j );
}
}  // namespace atlas::orca::meshgenerator