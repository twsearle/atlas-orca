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

CASE("test matchu between orca and regular ij indexing ") {
  std::string gridname = "ORCA2_T";

  //for (const std::string& gridname : gridnames) {}

  auto mypart = atlas::mpi::rank();
  auto orca_grid = OrcaGrid(gridname);

  StructuredGrid::YSpace yspace{grid::LinearSpacing{
      {-80., 90.}, orca_grid.ny(), true}};
  StructuredGrid::XSpace xspace{
      grid::LinearSpacing{{0., 360.}, orca_grid.nx(), false}};
  StructuredGrid regular_grid{xspace, yspace};

  SECTION("is the regular grid ij index unique?") {
    const idx_t size = regular_grid.size();
    std::set<gidx_t> ij_uid;
    for (gidx_t node = 0; node < size; ++node) {
      idx_t i, j;
      regular_grid.index2ij(node, i, j);
      ij_uid.insert(i*10*size + j);
    }

    if (size != ij_uid.size())
      std::cout << "[" << mypart
                << "] number of duplicate regular grid ij UID points "
                << size - ij_uid.size()
                << "/" << size << std::endl;
    EXPECT(size == ij_uid.size());
  }

  SECTION("are the orca grid internal ij indices unique?") {
    if (atlas::mpi::rank() == 0) {
      std::ofstream fold_file("northfold.csv");
      for (idx_t i = 0; i < orca_grid.nx(); ++i) {
        fold_file << std::setw(8) << i << ", ";
      }
      fold_file << std::endl;
      for (idx_t j = orca_grid.ny()+2; j > orca_grid.ny()-1; --j) {
        for (idx_t i = 0; i < orca_grid.nx(); ++i) {
          fold_file << std::setw(8) << orca_grid.periodicIndex(i, j) << ", ";
        }
        fold_file << std::endl;
      }
      fold_file.close();
    }
    const idx_t size = orca_grid.size();
    std::set<gidx_t> ij_uid;
    for (gidx_t node = 0; node < size; ++node) {
      idx_t i, j;
      orca_grid.index2ij(node, i, j);
      if (i >= orca_grid.nx() || i < 0) continue;
      if (j >= orca_grid.ny() || j < 0) continue;
      ij_uid.insert(i*10*size + j);
    }
    auto internal_size = orca_grid.nx()*orca_grid.ny();
    if (internal_size != ij_uid.size())
      std::cout << "[" <<  mypart
                << "] number of duplicate orca grid ij UID points "
                << internal_size - ij_uid.size()
                << "/" << internal_size << std::endl;
    EXPECT(internal_size == ij_uid.size());
  }

  SECTION("Are the boundary symmetries present in the orca grid ij indices?") {
    const idx_t size = orca_grid.size();
    atlas::PointLonLat lonlat, lonlat_halo;
    auto print_mismatch = [&](idx_t i, idx_t j) {
      std::cout << "[" << mypart
                << "] lonlat[0] " << lonlat[0] << " != " << lonlat_halo[0]
                << "\n lonlat[1] " << lonlat[1] << " != " << lonlat_halo[1]
                << "\n i, j " << i << ", " << j
                << "\n orca_grid.PeriodicIndex(i, j) " << orca_grid.periodicIndex(i, j)
                << std::endl;
    };
    for (gidx_t node = 0; node < size; ++node) {
      idx_t i, j;
      orca_grid.index2ij(node, i, j);
      if (j < 0) {
        continue;
      } else if (j == orca_grid.ny() ) {
        // check northfold boundary - first row
        lonlat = orca_grid.lonlat(i, j);
        auto pivot = orca_grid.nx() / 2;
        if (i == pivot) continue; // skip the complex case at the pivot.
        // count from right-hand-side of grid.
        lonlat_halo = orca_grid.lonlat(orca_grid.nx() - i, orca_grid.ny() + 2);
        if (orca_grid.periodicIndex(i, j) != orca_grid.periodicIndex(orca_grid.nx() - i - 1, orca_grid.ny() + 2)){
          std::cout << "------" << std::endl;
          print_mismatch(i,j);
          print_mismatch(orca_grid.nx() - i, orca_grid.ny() + 2);
          std::cout << "------" << std::endl;
        }
        EXPECT(orca_grid.periodicIndex(i, j) == orca_grid.periodicIndex(orca_grid.nx() - i - 1, orca_grid.ny() + 2));
      } else if (j == orca_grid.ny() + 1 ) {
        // check northfold boundary - centre fold row
        lonlat = orca_grid.lonlat(i, j);
      } else if (j == orca_grid.ny() + 2 ) {
        // check northfold boundary - second row
        lonlat = orca_grid.lonlat(i, j);
      } else if (i >= orca_grid.nx()) {
        // check east-west boundary
        lonlat = orca_grid.lonlat(i, j);
        lonlat_halo = orca_grid.lonlat(-i+orca_grid.nx(), j);
        EXPECT(lonlat[0] == lonlat_halo[0]);
        EXPECT(lonlat[1] == lonlat_halo[1]);
      }
    }
  }
}

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
