/*
 * (C) Copyright 2021- ECMWF.
 *
 * This software is licensed under the terms of the Apache Licence Version 2.0
 * which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
 * In applying this licence, ECMWF does not waive the privileges and immunities
 * granted to it by virtue of its status as an intergovernmental organisation
 * nor does it submit to any jurisdiction.
 */

#pragma once

#include <limits>
#include <memory>

#include "atlas/meshgenerator/MeshGenerator.h"
#include "atlas/util/Config.h"
#include "atlas/grid/Distribution.h"
#include "atlas-orca/grid/OrcaGrid.h"
#include "atlas-orca/meshgenerator/SurroundingRectangle.h"


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
class LocalOrcaGrid {
 public:
    struct Configuration {
        int nparts;
        int mypart;
        int halosize;
        int ny_orca_halo;
        int iy_glb_min;
        int iy_glb_max;
        int ix_glb_max;
        int ix_glb_min;
        Configuration() :
            nparts(std::numeric_limits<int>::lowest())
            , mypart(std::numeric_limits<int>::lowest())
            , halosize(std::numeric_limits<int>::lowest())
            , ny_orca_halo(std::numeric_limits<int>::lowest())
            , iy_glb_min(std::numeric_limits<int>::lowest())
            , iy_glb_max(std::numeric_limits<int>::lowest())
            , ix_glb_max(std::numeric_limits<int>::lowest())
            , ix_glb_min(std::numeric_limits<int>::lowest()) {}
        void check_consistency() const {
            const auto check = [&](const int value) {
                if (value == std::numeric_limits<int>::lowest())
                  eckit::BadParameter("atlas-orca/meshgenerator/LocalOrcaGrid: not all parameters set"); 
            };
            check(nparts);
            check(mypart);
            check(halosize);
            check(ny_orca_halo);
            check(iy_glb_min);
            check(iy_glb_max);
            check(ix_glb_max);
            check(ix_glb_min);
        }
    };

    std::vector<int> parts;
    std::vector<int> halo;
    std::vector<int> is_ghost;
    uint64_t size() const {return size_;}
    uint64_t halosize() const {return cfg_.halosize;}
    int ix_orca_min() const {return ix_orca_min_;}
    int ix_orca_max() const {return ix_orca_max_;}
    int iy_orca_min() const {return iy_orca_min_;}
    int iy_orca_max() const {return iy_orca_max_;}
    uint64_t nx_orca() const {return nx_orca_;}
    uint64_t ny_orca() const {return ny_orca_;}
    uint64_t nb_real_nodes() const {return nb_real_nodes_;}
    uint64_t nb_ghost_nodes() const {return nb_ghost_nodes_;}
    uint64_t nb_cells() const {return nb_cells_;}

    int index( int i, int j ) const;
    LocalOrcaGrid( const Grid& grid, const SurroundingRectangle& rectangle, const Configuration& cfg );
 private:
    const OrcaGrid orca_;
    const Configuration cfg_;
    uint64_t size_;
    int ix_orca_min_;
    int ix_orca_max_;
    int iy_orca_min_;
    int iy_orca_max_;
    uint64_t nx_orca_;
    uint64_t ny_orca_;
    uint64_t nb_real_nodes_;
    uint64_t nb_ghost_nodes_;
    uint64_t nb_cells_;
    std::vector<int> is_node_;
    std::vector<int> is_cell_;
};
}  // namespace meshgenerator
}  // namespace orca
}  // namespace atlas
