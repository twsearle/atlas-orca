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

/*
CASE("test surrounding rectangle ") {
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
      std::cout << "[" << cfg.mypart << "] " << regular_grid.type() << std::endl;

      EXPECT(regular_grid.ny() == grid.ny());
      for (idx_t ix = 0; ix < grid.ny(); ++ix) {
        EXPECT(regular_grid.nx(ix) == grid.nx());
      }
      std::cout << " last index? " << regular_grid.index(grid.nx()-1, grid.ny()-1) << std::endl;

      orca::meshgenerator::SurroundingRectangle rectangle(distribution, cfg);

      auto regular_mesh = atlas::Mesh(regular_grid, partitioner);

      auto fview_lonlat =
          array::make_view<double, 2>(regular_mesh.nodes().lonlat());
      auto fview_glb_idx =
          array::make_view<gidx_t, 1>(regular_mesh.nodes().global_index());

    std::ofstream nodeFile(std::string("is_node_") + distribution.type() + "-"
        + std::to_string(cfg.halosize) + "_p"
        + std::to_string(cfg.mypart) + ".csv");

    std::ofstream ghostFile(std::string("is_ghost_") + distribution.type() + "-"
        + std::to_string(cfg.halosize) + "_p"
        + std::to_string(cfg.mypart) + ".csv");

    std::ofstream haloFile(std::string("is_halo_") + distribution.type() + "-"
        + std::to_string(cfg.halosize) + "_p"
        + std::to_string(cfg.mypart) + ".csv");

    std::ofstream partFile(std::string("parts_") + distribution.type() + "-"
        + std::to_string(cfg.halosize) + "_p"
        + std::to_string(cfg.mypart) + ".csv");

      functionspace::NodeColumns regular_fs(regular_mesh);
      Field field_ghost = regular_fs.createField<int>(option::name("is_ghost"));
      auto fview_ghost = array::make_view<int, 1>(field_ghost);
      Field field_node = regular_fs.createField<int>(option::name("is_node"));
      auto fview_node = array::make_view<int, 1>(field_node);
      Field field_halo = regular_fs.createField<int>(option::name("is_halo"));
      auto fview_halo = array::make_view<int, 1>(field_halo);
      Field field_part = regular_fs.createField<int>(option::name("part_check"));
      auto fview_part = array::make_view<int, 1>(field_part);
      std::vector<int> indices;
      std::vector<bool> this_partition;
      for (uint64_t j = 0; j < rectangle.ny(); j++) {
        int iy_glb = rectangle.iy_min() + j;
        EXPECT(iy_glb < grid.ny());
        for (uint64_t i = 0; i < rectangle.nx(); i++) {
          int ix_glb = rectangle.ix_min() + i;
          EXPECT(ix_glb < grid.nx());
          auto ii = rectangle.index(i, j);
          indices.emplace_back(ii);

          haloFile  << i << ", " << j << ", " << rectangle.halo.at(ii) << std::endl;
          nodeFile  << i << ", " << j << ", " << (rectangle.is_node.at(ii) ? 1 : 0) << std::endl;
          ghostFile << i << ", " << j << ", " << rectangle.is_ghost.at(ii) << std::endl;
          partFile << i << ", " << j << ", " << rectangle.parts.at(ii) << std::endl;

          idx_t reg_grid_glb_idx  = regular_grid.index(ix_glb, iy_glb);
          idx_t orca_grid_glb_idx = grid.periodicIndex(ix_glb, iy_glb);
          idx_t reg_grid_remote_idx = 0;
          while(reg_grid_remote_idx < fview_glb_idx.size()) {
            if (reg_grid_glb_idx == fview_glb_idx(reg_grid_remote_idx))
              break;
            ++reg_grid_remote_idx;
          }

          if (rectangle.partition(ix_glb, iy_glb) == cfg.mypart) {
            this_partition.emplace_back(true);
          } else {
            this_partition.emplace_back(false);
          }

          if (reg_grid_remote_idx >= fview_halo.shape(0))
            std::cout << " reg_grid_remote_idx " << reg_grid_remote_idx << std::endl;
          //EXPECT(reg_grid_remote_idx < fview_halo.shape(0));

          //fview_halo(reg_grid_remote_idx) == 0;
          //if (rectangle.halo.at(ii) > 0) {
          //  fview_halo(regular_grid.index(ix_glb, iy_glb)) == 1;
          //  // all halo nodes should be ghost nodes
          //  //EXPECT(rectangle.is_ghost.at(ii));
          //}
          //fview_node(regular_grid.index(ix_glb, iy_glb)) == 0;
          //if (rectangle.is_node.at(ii)) {
          //  fview_node(regular_grid.index(ix_glb, iy_glb)) == 1;
          //}
          //fview_ghost(regular_grid.index(ix_glb, iy_glb)) == 0;
          //if (rectangle.is_ghost.at(ii)) {
          //  fview_ghost(regular_grid.index(ix_glb, iy_glb)) == 1;
          //}
          // If it is not a ghost node, it must be a node, however some ghost
          // nodes are also nodes.
          // TODO: Understand what is going on with this!
          if (!rectangle.is_ghost.at(ii)) {
            if (!rectangle.is_node.at(ii)) {
              std::cout << "[" << cfg.mypart << "] i " << i << " j " << j << " ii " << ii << std::endl;
            }
          }
        }
      }
      haloFile.close();
      nodeFile.close();
      ghostFile.close();
      partFile.close();

      int total_is_node =
          std::count(rectangle.is_node.begin(), rectangle.is_node.end(), true);
      int total_is_ghost =
          std::count(rectangle.is_ghost.begin(), rectangle.is_ghost.end(), true);
      EXPECT(total_is_node + total_is_ghost >= indices.size());

      EXPECT(indices.size() == rectangle.nx() * rectangle.ny());

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
                  << " ix_min " << rectangle.ix_min() << " ix_max "
                  << rectangle.ix_max() << " iy_min " << rectangle.iy_min()
                  << " iy_max " << rectangle.iy_max() << " indices.size() "
                  << indices.size() << " nx*ny " << rectangle.nx() * rectangle.ny()
                  << " number on this partition " << total_on_partition
                  << " number not on partition " << not_on_partition << std::endl;

        output::Gmsh gmsh(std::string("surroundingRect") +
                              std::to_string(cfg.nparts) + "_" + gridname + "_" +
                              distributionName + "_" + std::to_string(halo) +
                              ".msh",
                          Config("coordinates", "xy") | Config("info", true));
        gmsh.write(regular_mesh);
        gmsh.write(field_ghost);
        gmsh.write(field_node);
        gmsh.write(field_halo);
        gmsh.write(field_part);
      }

      if (cfg.nparts == 2) {
        if (cfg.mypart == 0) {
          EXPECT(rectangle.iy_min() == -1);
          EXPECT(rectangle.iy_max() == 147 + halo);
          EXPECT(rectangle.ix_min() == -1 - halo);
          EXPECT(rectangle.ix_max() == 90 + halo);
        }
        if (cfg.mypart == 1) {
          EXPECT(rectangle.iy_min() == -1);
          EXPECT(rectangle.iy_max() == 147 + halo);
          EXPECT(rectangle.ix_min() == 90 - halo);
          EXPECT(rectangle.ix_max() == 180 + halo);
        }
      }
      if (cfg.nparts == 4) {
        if (cfg.mypart == 0) {
          EXPECT(rectangle.iy_min() == -1);
          EXPECT(rectangle.iy_max() == 74 + halo);
          EXPECT(rectangle.ix_min() == -1 - halo);
          EXPECT(rectangle.ix_max() == 90 + halo);
        }
        if (cfg.mypart == 1) {
          EXPECT(rectangle.iy_min() == -1);
          EXPECT(rectangle.iy_max() == 74 + halo);
          EXPECT(rectangle.ix_min() == 89 - halo);
          EXPECT(rectangle.ix_max() == 180 + halo);
        }
        if (cfg.mypart == 2) {
          EXPECT(rectangle.iy_min() == 73 - halo);
          EXPECT(rectangle.iy_max() == 147 + halo);
          EXPECT(rectangle.ix_min() == -1 - halo);
          EXPECT(rectangle.ix_max() == 91 + halo);
        }
        if (cfg.mypart == 3) {
          EXPECT(rectangle.iy_min() == 73 - halo);
          EXPECT(rectangle.iy_max() == 147 + halo);
          EXPECT(rectangle.ix_min() == 90 - halo);
          EXPECT(rectangle.ix_max() == 180 + halo);
        }
      }
    }
  }
}
*/

} // namespace test
} // namespace atlas

int main(int argc, char **argv) { return atlas::test::run(argc, argv); }
