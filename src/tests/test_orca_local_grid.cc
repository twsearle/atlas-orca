/*
 * (C) Copyright 2021- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#include <algorithm>
#include <sstream>
#include <fstream>
#include <iomanip>

#include "atlas-orca/meshgenerator/SurroundingRectangle.h"
#include "atlas-orca/meshgenerator/LocalOrcaGrid.h"
#include "atlas/functionspace/NodeColumns.h"
#include "atlas/grid.h"
#include "atlas/grid/Spacing.h"
#include "atlas/grid/StructuredGrid.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/output/Gmsh.h"
#include "atlas/parallel/mpi/mpi.h"
#include "atlas/util/Config.h"
#include "atlas/util/function/VortexRollup.h"
#include "atlas/util/Point.h"

#include "atlas-orca/grid/OrcaGrid.h"
#include "atlas-orca/util/PointIJ.h"

#include "tests/AtlasTestEnvironment.h"

using Grid = atlas::Grid;
using Config = atlas::util::Config;

namespace atlas {
namespace test {

int wrap( idx_t value, idx_t lower, idx_t upper ) {
  // wrap around coordinate system when out of bounds
  const idx_t width = upper - lower;
  if (value < lower) {
    return wrap(value + width, lower, upper);
  }
  if (value > upper) {
    return wrap(value - width, lower, upper);
  }
  return value;
}

//-----------------------------------------------------------------------------

CASE("test surrounding local_orca ") {
  std::string gridname = "ORCA2_T";
  std::string distributionName = "checkerboard";

  auto rollup_plus = [](const double lon, const double lat) {
    return 1 + util::function::vortex_rollup(lon, lat, 0.0);
  };

  for (int halo = 0; halo < 1; ++halo) {
    SECTION(gridname + "_" + distributionName + "_halo" + std::to_string(halo)) {
      auto grid = OrcaGrid(gridname);
      auto partitioner_config = Config();
      partitioner_config.set("type", distributionName);
      auto partitioner = grid::Partitioner(partitioner_config);
      StructuredGrid::YSpace yspace{grid::LinearSpacing{
          {-80., 90.}, grid.ny(), true}};
      StructuredGrid::XSpace xspace{
          grid::LinearSpacing{{0., 360.}, grid.nx(), false}};
      StructuredGrid regular_grid{xspace, yspace};
      auto distribution = grid::Distribution(regular_grid, partitioner);

      orca::meshgenerator::SurroundingRectangle::Configuration cfg;
      cfg.mypart = mpi::rank();
      cfg.nparts = mpi::size();
      cfg.halosize = halo;
      cfg.nx_glb = grid.nx();
      cfg.ny_glb = grid.ny();
      orca::meshgenerator::SurroundingRectangle rectangle(distribution, cfg);
      std::cout << "[" << cfg.mypart << "] rectangle.ix_min " <<  rectangle.ix_min()
                << " rectangle.ix_max " <<  rectangle.ix_max()
                << " rectangle.iy_min " <<  rectangle.iy_min()
                << " rectangle.iy_max " <<  rectangle.iy_max() << std::endl;
      orca::meshgenerator::LocalOrcaGrid local_orca(grid, rectangle);

      std::vector<int> indices;
      std::vector<bool> this_partition;
      for (uint64_t j = 0; j < local_orca.ny(); j++) {
        int iy_glb = local_orca.iy_min() + j;
        EXPECT(iy_glb < grid.ny() + grid.haloNorth() + grid.haloSouth());
        for (uint64_t i = 0; i < local_orca.nx(); i++) {
          int ix_glb = local_orca.ix_min() + i;
          EXPECT(ix_glb < grid.nx() + grid.haloWest() + grid.haloEast());
          auto ii = local_orca.index(i, j);
          indices.emplace_back(ii);

          idx_t reg_grid_glb_idx  = regular_grid.index(ix_glb, iy_glb);
          idx_t orca_grid_glb_idx = grid.periodicIndex(ix_glb, iy_glb);
          idx_t reg_grid_remote_idx = 0;

          if (local_orca.parts.at(ii) == cfg.mypart) {
            this_partition.emplace_back(true);
          } else {
            this_partition.emplace_back(false);
          }
        }
      }
      int total_is_node =
          std::count(local_orca.is_node.begin(), local_orca.is_node.end(), true);
      int total_is_ghost =
          std::count(local_orca.is_ghost.begin(), local_orca.is_ghost.end(), true);
      EXPECT(total_is_node + total_is_ghost >= indices.size());
      EXPECT(indices.size() == local_orca.nx() * local_orca.ny());

      {
        // diagnostics
        auto total_on_partition =
            std::count(this_partition.begin(), this_partition.end(), true);
        auto not_on_partition =
            std::count(this_partition.begin(), this_partition.end(), false);

        std::cout << "[" << cfg.mypart << "] grid.haloWest() " << grid.haloWest()
                  << " grid.haloEast() " << grid.haloEast()
                  << " grid.haloNorth() " << grid.haloNorth()
                  << " grid.haloSouth() " << grid.haloSouth()
                  << std::endl;
        std::cout << "[" << cfg.mypart << "]"
                  << " ix_orca_min " << local_orca.ix_min() << " ix_orca_max "
                  << local_orca.ix_max() << " iy_orca_min " << local_orca.iy_min()
                  << " iy_orca_max " << local_orca.iy_max() << " indices.size() "
                  << indices.size() << " nx*ny " << local_orca.nx() * local_orca.ny()
                  << " number on this partition " << total_on_partition
                  << " number not on partition " << not_on_partition << std::endl;

        output::Gmsh gmsh(std::string("surroundingRect") +
                              std::to_string(cfg.nparts) + "_" + gridname + "_" +
                              distributionName + "_" + std::to_string(halo) +
                              ".msh",
                          Config("coordinates", "xy") | Config("info", true));
      }

      if (cfg.nparts == 2) {
        if (cfg.mypart == 0) {
          EXPECT(local_orca.iy_min() == -1);
          EXPECT(local_orca.iy_max() == 147 + halo);
          EXPECT(local_orca.ix_min() == -1 - halo);
          EXPECT(local_orca.ix_max() == 89 + halo);
          EXPECT(local_orca.nx() == (89+1) + 1 + 2*halo);
          EXPECT(local_orca.ny() == 149 + 3*halo);
        }
        if (cfg.mypart == 1) {
          EXPECT(local_orca.iy_min() == -1);
          EXPECT(local_orca.iy_max() == 147 + halo);
          EXPECT(local_orca.ix_min() == 90 - halo);
          EXPECT(local_orca.ix_max() == 180 + halo);
          EXPECT(local_orca.nx() == (180-90) + 1 + 2*halo);
          EXPECT(local_orca.ny() == 149 + 3*halo);
        }
      }
    }
  }
}

} // namespace test
} // namespace atlas

int main(int argc, char **argv) { return atlas::test::run(argc, argv); }
