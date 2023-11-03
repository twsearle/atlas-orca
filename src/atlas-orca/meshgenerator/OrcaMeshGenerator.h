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

#include "atlas/meshgenerator/MeshGenerator.h"
#include "atlas/meshgenerator/detail/MeshGeneratorImpl.h"
#include "atlas/util/Config.h"

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace eckit {
class Parametrisation;
}

namespace atlas {
class Mesh;
class OrcaGrid;
}  // namespace atlas

namespace atlas {
namespace grid {
class Distribution;
}  // namespace grid
}  // namespace atlas
#endif

namespace atlas {
namespace orca {
namespace meshgenerator {

// BAD CHECKERBOARD ORCA2_T MPI 2 POINTS
//const int64_t TEST_MASTER_GLOBAL_IDX = 26664;
//const int64_t TEST_MASTER_GLOBAL_IDX = 26580;
//const int64_t TEST_MASTER_GLOBAL_IDX = 26575;
//const int64_t TEST_MASTER_GLOBAL_IDX = 26757;
const int64_t TEST_MASTER_GLOBAL_IDX = -1;
//
const int64_t TEST_REMOTE_IDX = 13287;
const int64_t TEST_PARTITION = -1;


//----------------------------------------------------------------------------------------------------------------------

class OrcaMeshGenerator : public MeshGenerator::Implementation {
public:
    OrcaMeshGenerator( const eckit::Parametrisation& = util::NoConfig() );

    using MeshGenerator::Implementation::generate;

    void generate( const Grid&, const grid::Partitioner&, Mesh& ) const override;
    void generate( const Grid&, const grid::Distribution&, Mesh& ) const override;
    void generate( const Grid&, Mesh& ) const override;

    std::string type() const override { return "orca"; }
    static std::string static_type() { return "orca"; }

private:
    void hash( eckit::Hash& ) const override;
    static void build_remote_index(Mesh& mesh);

    bool include_pole_{false};
    bool fixup_{true};
    int nparts_;
    int mypart_;
};

//----------------------------------------------------------------------------------------------------------------------

}  // namespace meshgenerator
}  // namespace orca
}  // namespace atlas
