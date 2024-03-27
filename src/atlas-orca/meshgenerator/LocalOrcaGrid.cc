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

#include "atlas-orca/meshgenerator/LocalOrcaGrid.h"


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace eckit {
class Parametrisation;
}

namespace atlas {
class OrcaGrid;
}  // namespace atlas

#endif

namespace atlas {
namespace orca {
namespace meshgenerator {

//----------------------------------------------------------------------------------------------------------------------
LocalOrcaGrid::LocalOrcaGrid( const Grid& grid, const SurroundingRectangle& rectangle, const Configuration& cfg ) :
        orca_(grid), cfg_(cfg) {
  if (rectangle.ix_min() <= 0) {
    ix_orca_min_ = rectangle.ix_min() - orca_.haloWest();
  }
  if (rectangle.ix_max() >= orca_.nx()) {
    ix_orca_max_ = rectangle.ix_max() + orca_.haloEast();
  }
  if (rectangle.iy_min() <= 0) {
    iy_orca_min_ = rectangle.iy_min() - orca_.haloSouth();
  }
  if (rectangle.iy_max() >= orca_.ny()) {
    iy_orca_max_ = rectangle.iy_max() + orca_.haloNorth();
  }

  // dimensions of the rectangle including the ORCA halo points
  nx_orca_ = ix_orca_max_ - ix_orca_min_;
  ny_orca_ = iy_orca_max_ - iy_orca_min_;

  // partitions and local indices in SR
  parts.resize( size_, -1 );
  halo.resize( size_, 0 );
  is_node.resize( size_, false );
  is_ghost.resize( size_, true );
  {
    //atlas_omp_parallel_for( idx_t iy = 0; iy < ny_; iy++ )
    for( size_t iy = 0; iy < ny_orca_; iy++ ) {
      for ( size_t ix = 0; ix < nx_orca_; ix++ ) {
        // TODO: identify partition based on rectangle partition + orca info
        // TODO: identify halo based on halo partition + orca info
        bool halo_found = false;
        idx_t ii = index( ix, iy );
        idx_t reg_ii = 0;
        if (ix < rectangle.nx() && iy < rectangle.ny()) {
          reg_ii = rectangle.index(ix, iy);
        } else if (ix < rectangle.nx()) {
          reg_ii = rectangle.index(ix, rectangle.ny()-1);
        } else {
          reg_ii = rectangle.index(rectangle.nx()-1, iy);
        }
        parts.at( ii )    = rectangle.parts.at( reg_ii );
        halo.at( ii )     = rectangle.halo.at( reg_ii );
        is_node.at( ii )  = rectangle.is_node.at( reg_ii );
        is_ghost.at( ii ) = rectangle.is_ghost.at( reg_ii );
        int halo_dist = cfg_.halosize;
        if ((cfg_.halosize > 0) && parts.at( ii ) != cfg_.mypart ) {
          if (halo_found) {
            halo.at( ii ) = halo_dist;
          }
        }

        is_ghost.at( ii ) = ( parts.at( ii ) != cfg_.mypart );
      }
    }
  }

  // determine number of cells and number of nodes
  uint16_t nb_halo_nodes = 0;
  {  // Compute SR.is_node
    std::vector<int> is_cell(size_, false);
    auto mark_node_used = [&]( int ix, int iy ) {
      idx_t ii = index( ix, iy );
      if ( !is_node.at(ii) ) {
        ++nb_real_nodes_;
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
    // Loop over all elements in rectangle
    nb_cells_ = 0;
    nb_real_nodes_ = 0;
    nb_ghost_nodes_ = 0;
    for ( idx_t iy = 0; iy < ny_orca_-1; iy++ ) {
      for ( idx_t ix = 0; ix < nx_orca_-1; ix++ ) {
        if ( is_ghost[index( ix, iy )]) {
          ++nb_ghost_nodes_;
          if ( halo[index( ix, iy )] != 0)
            ++nb_halo_nodes;
        } else {
          mark_cell_used( ix, iy );
          mark_node_used( ix, iy );
          mark_node_used( ix + 1, iy );
          mark_node_used( ix + 1, iy + 1 );
          mark_node_used( ix, iy + 1 );
        } // if not ghost index
      }
    }
  }
}
int LocalOrcaGrid::index( int i, int j ) const {
  ATLAS_ASSERT_MSG(static_cast<size_t>(i) < nx_orca_,
     std::string("i >= nx_orca_: ") + std::to_string(i) + " >= " + std::to_string(nx_orca_));
  ATLAS_ASSERT_MSG(static_cast<size_t>(j) < ny_orca_,
    std::string("j >= ny_orca_: ") + std::to_string(j) + " >= " + std::to_string(ny_orca_));
  return j * nx_orca_ + i;
}
}  // namespace meshgenerator
}  // namespace orca
}  // namespace atlas
