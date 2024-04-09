/*
 * (C) Copyright 2021- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include "SurroundingRectangle.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <fstream>
#include <iomanip>

#include "atlas/array/Array.h"
#include "atlas/field/Field.h"
#include "atlas/grid/Distribution.h"
#include "atlas/grid/Partitioner.h"
#include "atlas/parallel/mpi/mpi.h"
#include "atlas/runtime/Exception.h"
#include "atlas/runtime/Log.h"
#include "atlas/util/Constants.h"
#include "atlas/util/CoordinateEnums.h"
#include "atlas/util/Geometry.h"


namespace atlas::orca::meshgenerator {

namespace {
int wrap( idx_t value, idx_t lower, idx_t upper ) {
  // wrap around coordinate system when out of bounds
  const idx_t width = upper - lower;
  if (value < lower) {
    return wrap(value + width, lower, upper);
  }
  if (value >= upper) {
    return wrap(value - width, lower, upper);
  }
  return value;
}
}  // namespace


int SurroundingRectangle::index( int i, int j ) const {
  ATLAS_ASSERT_MSG(i < nx_, std::string("i >= nx_: ") + std::to_string(i) + " >= " + std::to_string(nx_));
  ATLAS_ASSERT_MSG(j < ny_, std::string("j >= ny_: ") + std::to_string(j) + " >= " + std::to_string(ny_));
  return j * nx_ + i;
}

int SurroundingRectangle::partition( idx_t ix, idx_t iy ) const {
  ATLAS_ASSERT_MSG(ix < cfg_.nx_glb, std::string("ix >= cfg_.nx_glb: ") + std::to_string(ix) + " >= " + std::to_string(cfg_.nx_glb));
  ATLAS_ASSERT_MSG(iy < cfg_.ny_glb, std::string("iy >= cfg_.ny_glb: ") + std::to_string(iy) + " >= " + std::to_string(cfg_.ny_glb));
  return distribution_.partition( iy * cfg_.nx_glb + ix );
}

SurroundingRectangle::SurroundingRectangle(
    const grid::Distribution& distribution,
    const Configuration& cfg )
  : distribution_(distribution), cfg_(cfg) {
  ATLAS_TRACE();
  cfg_.check_consistency();

  std::ofstream logFile(distribution.type() + "-"
      + std::to_string(cfg_.halosize) + "_p"
      + std::to_string(cfg_.mypart) + ".log");

  // determine rectangle (ix_min_:ix_max_) x (iy_min_:iy_max_) surrounding the nodes on this processor
  ix_min_         = cfg_.nx_glb;
  ix_max_         = 0;
  iy_min_         = cfg_.ny_glb;
  iy_max_         = 0;
  nb_real_nodes_owned_by_rectangle = 0;

  // TODO: These "bounds"  are on the imaginary wrapped rectangle including halo and periodic points.
  // points out of bounds of the reglatlon grid are either in the orca halo points, or are in the halo
  {
    ATLAS_TRACE( "find rectangle bounds" );
    atlas_omp_parallel {
      int ix_min_TP = ix_min_;
      int ix_max_TP = ix_max_;
      int iy_min_TP = iy_min_;
      int iy_max_TP = iy_max_;
      int nb_real_nodes_owned_by_rectangle_TP = 0;
      atlas_omp_for( idx_t iy = 0; iy < cfg_.ny_glb; iy++ ) {
        for ( idx_t ix = 0; ix < cfg_.nx_glb; ix++ ) {
          // wrap around coordinate system when out of bounds on the rectangle
          int ix_wrapped = wrap(ix, 0, cfg_.nx_glb);
          int iy_wrapped = wrap(iy, 0, cfg_.ny_glb);
          int p = partition( ix_wrapped, iy_wrapped );
          if ( p == cfg_.mypart ) {
            ix_min_TP = std::min<idx_t>( ix_min_TP, ix );
            ix_max_TP = std::max<idx_t>( ix_max_TP, ix );
            iy_min_TP = std::min<idx_t>( iy_min_TP, iy );
            iy_max_TP = std::max<idx_t>( iy_max_TP, iy );
            nb_real_nodes_owned_by_rectangle_TP++;
          } else if (cfg_.halosize > 0) {
            // use lambda to break from both loops at once
            [&] {
              for (idx_t dhx = -cfg_.halosize; dhx < cfg_.halosize + 1; ++dhx) {
                for (idx_t dhy = -cfg_.halosize; dhy < cfg_.halosize + 1; ++dhy) {
                  if ((dhy == 0 && dhx == 0))
                    continue;
                  // wrap around coordinate system when out of bounds on the rectangle
                  int hx = wrap(ix + dhx, 0, cfg_.nx_glb);
                  int hy = wrap(iy + dhy, 0, cfg_.ny_glb);
                  int p_halo = partition( hx, hy );

                  if ( p_halo == cfg_.mypart ) {
                    //nb_real_nodes_owned_by_rectangle_TP++;
                    if (ix_max_TP < ix) logFile << "[" << cfg_.mypart << "] ix_max bumped by halo: hx " << hx  << " ix " << ix << " ix_max_TP " << ix_max_TP << std::endl;
                    iy_min_TP = std::min<idx_t>( iy_min_TP, iy );
                    iy_max_TP = std::max<idx_t>( iy_max_TP, iy );
                    // NOTE. We need to update the max if we have wrapped
                    // around the grid to the left, or the min if we have
                    // wrapped around the grid to the right.
                    if (ix + dhx < 0) {
                      ix_max_TP = std::max<idx_t>( ix_max_TP, ix + cfg_.nx_glb );
                    } else if (ix + dhx >= cfg_.nx_glb) {
                      ix_min_TP = std::min<idx_t>( ix_min_TP, ix - cfg_.nx_glb );
                    } else {
                      ix_min_TP = std::min<idx_t>( ix_min_TP, ix );
                      ix_max_TP = std::max<idx_t>( ix_max_TP, ix );
                    }
                    return;
                  }
                }
              }
            }();
          }
          atlas_omp_critical {
            nb_real_nodes_owned_by_rectangle += nb_real_nodes_owned_by_rectangle_TP;
            ix_min_ = std::min<int>( ix_min_TP, ix_min_);
            ix_max_ = std::max<int>( ix_max_TP, ix_max_);
            iy_min_ = std::min<int>( iy_min_TP, iy_min_);
            iy_max_ = std::max<int>( iy_max_TP, iy_max_);
          }
        }
      }
    }
  }

  // dimensions of the surrounding rectangle
  nx_ = ix_max_ - ix_min_;
  ny_ = iy_max_ - iy_min_;

  logFile << "[" << cfg_.mypart << "] ix_min: "     << ix_min_ << std::endl;
  logFile << "[" << cfg_.mypart << "] ix_max: "     << ix_max_ << std::endl;
  logFile << "[" << cfg_.mypart << "] iy_min: "     << iy_min_ << std::endl;
  logFile << "[" << cfg_.mypart << "] iy_max: "     << iy_max_ << std::endl;

  // upper estimate for number of nodes
  uint64_t size = ny_ * nx_;

  // partitions and local indices in SR
  parts.resize( size, -1 );
  halo.resize( size, 0 );
  is_ghost.resize( size, true );
  // vectors marking nodes that are necessary for this proc's cells

  {
    ATLAS_TRACE( "partition, is_ghost, halo" );
    //atlas_omp_parallel_for( idx_t iy = 0; iy < ny_; iy++ )
    for( idx_t iy = 0; iy < ny_; iy++ ) {
      idx_t iy_reg_glb = iy_min_ + iy;  // global y-index in reg-grid-space
      for ( idx_t ix = 0; ix < nx_; ix++ ) {
        idx_t ii         = index( ix, iy );
        idx_t ix_reg_glb = ix_min_ + ix;  // global x-index in reg-grid-space
        parts.at( ii ) = partition( ix_reg_glb, iy_reg_glb );
        bool halo_found = false;
        int halo_dist = cfg_.halosize;
        if ((cfg_.halosize > 0) && parts.at( ii ) != cfg_.mypart ) {
          // search the surrounding halosize index square for a node on my
          // partition to determine the halo distance
          for (idx_t dhy = -cfg_.halosize; dhy <= cfg_.halosize; ++dhy) {
            for (idx_t dhx = -cfg_.halosize; dhx <= cfg_.halosize; ++dhx) {
              if (dhx == 0 && dhy == 0) continue;
              // wrap around coordinate system when out of bounds on the rectangle
              const int hx = wrap(ix_reg_glb + dhx, 0, cfg_.nx_glb);
              const int hy = wrap(iy_reg_glb + dhy, 0, cfg_.ny_glb);
              if (partition(hx, hy) == cfg_.mypart) {
                // find the minimum distance from this halo node to
                // a node on the partition
                auto dist = std::max(std::abs(dhx), std::abs(dhy));
                halo_dist = std::min(dist, halo_dist);
                halo_found = true;
              }
            }
          }
          if (halo_found) {
            halo.at( ii ) = halo_dist;
          }
        }

        is_ghost.at( ii ) = ( parts.at( ii ) != cfg_.mypart );
      }
    }
  }
  logFile << std::setw(5) << std::setfill('0');
  logFile << "[" << cfg_.mypart << "] nx                      = " << nx_ << std::endl;
  logFile << "[" << cfg_.mypart << "] ny                      = " << ny_ << std::endl;
  logFile << "[" << cfg_.mypart << "] halosize                = " << cfg_.halosize << std::endl;
  logFile << "[" << cfg_.mypart << "] ny * nx_                = " << ny_ * nx_ << std::endl;
  logFile << "[" << cfg_.mypart << "] ny * (nx_ + 2*halosize) = " << ny_ * (nx_ + 2*cfg_.halosize) << std::endl;
  logFile << "[" << cfg_.mypart << "] nb_real_nodes_owned_by_rectangle = " << nb_real_nodes_owned_by_rectangle << std::endl;
  logFile << "[" << cfg_.mypart << "] end of SR output" << std::endl;
  logFile.close();
}

}  // namespace atlas::orca::meshgenerator 
