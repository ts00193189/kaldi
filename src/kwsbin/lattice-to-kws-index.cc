// kwsbin/lattice-to-kws-index.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
//                 Lucas Ondel

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-utils.h"
#include "lat/kaldi-lattice.h"
#include "kws/kaldi-kws.h"
#include "kws/kws-functions.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using fst::VectorFst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;

    const char *usage =
        "Create an inverted index of the given lattices. The output index is \n"
        "in the T*T*T semiring. For details for the semiring, please refer to\n"
        "Dogan Can and Murat Saraclar's paper named "
        "\"Lattice Indexing for Spoken Term Detection\"\n"
        "\n"
        "Usage: lattice-to-kws-index [options]  "
        " <utter-symtab-rspecifier> <lattice-rspecifier> <index-wspecifier>\n"
        "e.g.: \n"
        " lattice-to-kws-index ark:utter.symtab ark:1.lats ark:global.idx\n";

    ParseOptions po(usage);

    int32 frame_subsampling_factor = 1;
    int32 max_silence_frames = 50;
    bool strict = true;
    bool allow_partial = true;
    BaseFloat max_states_scale = 4;
    po.Register("frame-subsampling-factor", &frame_subsampling_factor,
                "Frame subsampling factor. (Default value 1)");
    po.Register("max-silence-frames", &max_silence_frames,
                "If --frame-subsampling-factor is used, --max-silence-frames "
                "is relative to the the input, not the output frame rate "
                "(we divide by frame-subsampling-factor and round to "
                "the closest integer, to get the number of symbols in the "
                "lattice).");
    po.Register("strict", &strict, "Setting --strict=false will cause "
                "successful termination even if we processed no lattices.");
    po.Register("max-states-scale", &max_states_scale, "Number of states in the"
                " original lattice times this scale is the number of states "
                "allowed when optimizing the index. Negative number means no "
                "limit on the number of states.");
    po.Register("allow-partial", &allow_partial, "Allow partial output if fails"
                " to determinize, otherwise skip determinization if it fails.");

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    max_silence_frames = 0.5 +
            max_silence_frames / static_cast<float>(frame_subsampling_factor);
    std::string usymtab_rspecifier = po.GetOptArg(1),
        lats_rspecifier = po.GetArg(2),
        index_wspecifier = po.GetArg(3);

    // We use RandomAccessInt32Reader to read the utterance symtab table.
    RandomAccessInt32Reader usymtab_reader(usymtab_rspecifier);

    // We read the lattice in as CompactLattice; We need the CompactLattice
    // structure for the rest of the work
    SequentialCompactLatticeReader clat_reader(lats_rspecifier);

    TableWriter< fst::VectorFstTplHolder<KwsLexicographicArc> >
                                                index_writer(index_wspecifier);

    int32 n_done = 0;
    int32 n_fail = 0;

    int32 max_states = -1;

    for (; !clat_reader.Done(); clat_reader.Next()) {
      std::string key = clat_reader.Key();
      CompactLattice clat = clat_reader.Value();
      clat_reader.FreeCurrent();
      KALDI_LOG << "Processing lattice " << key;

      if (max_states_scale > 0) {
        max_states = static_cast<int32>(
            max_states_scale * static_cast<BaseFloat>(clat.NumStates()));
      }

      // Check if we have the corresponding utterance id.
      if (!usymtab_reader.HasKey(key)) {
        KALDI_WARN << "Cannot find utterance id for " << key;
        n_fail++;
        continue;
      }
      int32 utterance_id = usymtab_reader.Value(key);

      // Topologically sort the lattice, if not already sorted.
      uint64 props = clat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&clat) == false) {
          KALDI_WARN << "Cycles detected in lattice " << key;
          n_fail++;
          continue;
        }
      }

      // Construct KWS index.
      KwsLexicographicFst index_transducer;
      if (!LatticeToKwsIndexDestructive(&clat, utterance_id, max_silence_frames,
                                        max_states, allow_partial,
                                        &index_transducer)) {
        KALDI_WARN << "KWS index construction failed for lattice " << key;
        n_fail++;
        continue;
      }

      // Write result
      index_writer.Write(key, index_transducer);

      n_done++;
    }

    KALDI_LOG << "Done " << n_done << " lattices, failed for " << n_fail;
    if (strict == true)
      return (n_done != 0 ? 0 : 1);
    else
      return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
