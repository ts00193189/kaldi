// kws/kws-functions.cc

// Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)

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


#include "lat/lattice-functions.h"
#include "kws/kws-functions.h"
#include "fstext/determinize-star.h"
#include "fstext/epsilon-property.h"

// this file implements things in kws-functions.h; it's an overflow from
// kws-functions.cc (we split it up for compilation speed and to avoid
// generating too-large object files on cygwin).

namespace kaldi {


// This function replaces a symbol with epsilon wherever it appears
// (fst must be an acceptor).
template<class Arc>
static void ReplaceSymbolWithEpsilon(typename Arc::Label symbol,
                                     fst::VectorFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;
  for (StateId s = 0; s < fst->NumStates(); s++) {
    for (fst::MutableArcIterator<fst::VectorFst<Arc> > aiter(fst, s);
         !aiter.Done(); aiter.Next()) {
      Arc arc = aiter.Value();
      KALDI_ASSERT(arc.ilabel == arc.olabel);
      if (arc.ilabel == symbol) {
        arc.ilabel = 0;
        arc.olabel = 0;
        aiter.SetValue(arc);
      }
    }
  }
}

void DoFactorMerging(KwsProductFst *factor_transducer,
                     KwsLexicographicFst *index_transducer) {
  using namespace fst;
  typedef KwsProductFst::Arc::Label Label;

  // Encode the transducer first
  EncodeMapper<KwsProductArc> encoder(kEncodeLabels, ENCODE);
  Encode(factor_transducer, &encoder);


  // We want DeterminizeStar to remove epsilon arcs, so turn whatever it encoded
  // epsilons as, into actual epsilons.
  {
    KwsProductArc epsilon_arc(0, 0, KwsProductWeight::One(), 0);
    Label epsilon_label = encoder(epsilon_arc).ilabel;
    ReplaceSymbolWithEpsilon(epsilon_label, factor_transducer);
  }


  MaybeDoSanityCheck(*factor_transducer);

  // Use DeterminizeStar
  KALDI_VLOG(2) << "DoFactorMerging: determinization...";
  KwsProductFst dest_transducer;
  DeterminizeStar(*factor_transducer, &dest_transducer);

  MaybeDoSanityCheck(dest_transducer);

  // Commenting the minimization out, as it moves states/arcs in a way we don't
  // want in some rare cases. For example, if we have two arcs from starting
  // state, which have same words on the input side, but different cluster IDs
  // on the output side, it may make the two arcs sharing a common final arc,
  // which will cause problem in the factor disambiguation stage (we will not
  // be able to add disambiguation symbols for both paths). We do a final step
  // optimization anyway so commenting this out shouldn't matter too much.
  // KALDI_VLOG(2) << "DoFactorMerging: minimization...";
  // Minimize(&dest_transducer);

  MaybeDoSanityCheck(dest_transducer);

  Decode(&dest_transducer, encoder);

  Map(dest_transducer, index_transducer, KwsProductFstToKwsLexicographicFstMapper());
}

void DoFactorDisambiguation(KwsLexicographicFst *index_transducer) {
  using namespace fst;
  typedef KwsLexicographicArc::StateId StateId;

  StateId ns = index_transducer->NumStates();
  for (StateId s = 0; s < ns; s++) {
    for (MutableArcIterator<KwsLexicographicFst>
         aiter(index_transducer, s); !aiter.Done(); aiter.Next()) {
      KwsLexicographicArc arc = aiter.Value();
      if (index_transducer->Final(arc.nextstate) != KwsLexicographicWeight::Zero())
        arc.ilabel = s;
      else
        arc.olabel = 0;
      aiter.SetValue(arc);
    }
  }
}

void OptimizeFactorTransducer(KwsLexicographicFst *index_transducer,
                              int32 max_states,
                              bool allow_partial) {
  using namespace fst;
  KwsLexicographicFst ifst = *index_transducer;
  EncodeMapper<KwsLexicographicArc> encoder(kEncodeLabels, ENCODE);
  Encode(&ifst, &encoder);
  KALDI_VLOG(2) << "OptimizeFactorTransducer: determinization...";
  if (allow_partial) {
    DeterminizeStar(ifst, index_transducer, kDelta, NULL, max_states, true);
  } else {
      try {
        DeterminizeStar(ifst, index_transducer, kDelta, NULL, max_states,
                        false);
      } catch(const std::exception &e) {
        KALDI_WARN << e.what();
        *index_transducer = ifst;
      }
  }
  KALDI_VLOG(2) << "OptimizeFactorTransducer: minimization...";
  Minimize(index_transducer, static_cast<KwsLexicographicFst *>(NULL), fst::kDelta, true);
  Decode(index_transducer, encoder);
}

bool LatticeToKwsIndex(const CompactLattice &clat,
                       int32 utterance_id,
                       int32 max_silence_frames,
                       int32 max_states,
                       bool allow_partial,
                       KwsLexicographicFst *index_transducer) {
    CompactLattice temp(clat);
    return LatticeToKwsIndexDestructive(&temp, utterance_id, max_silence_frames,
                                        max_states, allow_partial,
                                        index_transducer);
}

bool LatticeToKwsIndexDestructive(CompactLattice *clat,
                                  int32 utterance_id,
                                  int32 max_silence_frames,
                                  int32 max_states,
                                  bool allow_partial,
                                  KwsLexicographicFst *index_transducer) {
  // Get the alignments
  std::vector<int32> state_times;
  CompactLatticeStateTimes(*clat, &state_times);

  // Cluster the arcs in the CompactLattice, write the cluster_id on the
  // output label side.
  // ClusterLattice() corresponds to the second part of the preprocessing in
  // Can and Saraclar's paper -- clustering. Note that we do the first part
  // of preprocessing (the weight pushing step) later when generating the
  // factor transducer.
  KALDI_VLOG(1) << "Arc clustering...";
  if (!ClusterLattice(clat, state_times)) {
    KALDI_WARN << "State id's and alignments do not match";
    return false;
  }

  // The next part is something new, not in the Can and Saraclar paper.  It is
  // necessary because we have epsilon arcs, due to silences, in our
  // lattices.  We modify the factor transducer, while maintaining
  // equivalence, to ensure that states don't have both epsilon *and*
  // non-epsilon arcs entering them.  (and the same, with "entering"
  // replaced with "leaving").  Later we will find out which states have
  // non-epsilon arcs leaving/entering them and use it to be more selective
  // in adding arcs to connect them with the initial/final states.  The goal
  // here is to disallow silences at the beginning or ending of a keyword
  // occurrence.
  if (true) {
    EnsureEpsilonProperty(clat);
    fst::TopSort(clat);
    // We have to recompute the state times because they will have changed.
    CompactLatticeStateTimes(*clat, &state_times);
  }

  // Generate factor transducer
  // CreateFactorTransducer() corresponds to the "Factor Generation" part of
  // Can and Saraclar's paper. But we also move the weight pushing step to
  // this function as we have to compute the alphas and betas anyway.
  KALDI_VLOG(1) << "Generating factor transducer...";
  KwsProductFst factor_transducer;
  if (!CreateFactorTransducer(*clat, state_times, utterance_id,
                              &factor_transducer)) {
    KALDI_WARN << "Cannot generate factor transducer";
    return false;
  }

  MaybeDoSanityCheck(factor_transducer);

  // Remove long silence arc
  // We add the filtering step in our implementation. This is because gap
  // between two successive words in a query term should be less than 0.5s
  if (max_silence_frames >= 0) {
    KALDI_VLOG(1) << "Removing long silence...";
    RemoveLongSilences(max_silence_frames, state_times, &factor_transducer);
    MaybeDoSanityCheck(factor_transducer);
  }

  // Do factor merging, and return a transducer in T*T*T semiring. This step
  // corresponds to the "Factor Merging" part in Can and Saraclar's paper.
  KALDI_VLOG(1) << "Merging factors...";
  DoFactorMerging(&factor_transducer, index_transducer);

  MaybeDoSanityCheck(*index_transducer);

  // Do factor disambiguation. It corresponds to the "Factor Disambiguation"
  // step in Can and Saraclar's paper.
  KALDI_VLOG(1) << "Doing factor disambiguation...";
  DoFactorDisambiguation(index_transducer);

  MaybeDoSanityCheck(*index_transducer);

  // Optimize the above factor transducer. It corresponds to the
  // "Optimization" step in the paper.
  KALDI_VLOG(1) << "Optimizing factor transducer...";
  OptimizeFactorTransducer(index_transducer, max_states, allow_partial);

  MaybeDoSanityCheck(*index_transducer);

  return true;
}

void OptimizeKwsIndex(KwsLexicographicFst *index, int32 max_states) {
  using namespace fst;
  KwsLexicographicFst ifst = *index;
  EncodeMapper<KwsLexicographicArc> encoder(kEncodeLabels, ENCODE);
  Encode(&ifst, &encoder);
  try {
    DeterminizeStar(ifst, index, kDelta, NULL, max_states);
  } catch(const std::exception &e) {
    KALDI_WARN << e.what()
               << " (should affect speed of search but not results)";
    *index = ifst;
  }
  Minimize(index, static_cast<KwsLexicographicFst*>(NULL), kDelta, true);
  Decode(index, encoder);
}

void EncodeKwsDisambiguationSymbols(
    KwsLexicographicFst *index,
    fst::internal::EncodeTable<KwsLexicographicArc> *encode_table) {
  using namespace fst;
  for (StateIterator<KwsLexicographicFst> siter(*index);
                                         !siter.Done(); siter.Next()) {
    for (MutableArcIterator<KwsLexicographicFst>
         aiter(index, siter.Value()); !aiter.Done(); aiter.Next()) {
      KwsLexicographicArc arc = aiter.Value();
      // Skip the non-final arcs
      if (index->Final(arc.nextstate) == KwsLexicographicWeight::Zero())
        continue;
      // Encode the input and output label of the final arc, and this is the
      // new output label for this arc; set the input label to <epsilon>
      arc.olabel = encode_table->Encode(arc);
      arc.ilabel = 0;
      aiter.SetValue(arc);
    }
  }
  ArcSort(index, ILabelCompare<KwsLexicographicArc>());
}

bool SearchKwsIndex(const KwsLexicographicFst &index,
                    const fst::StdVectorFst &keyword,
                    const fst::internal::EncodeTable<KwsLexicographicArc> &encode_table,
                    int32 n_best,
                    std::vector<std::tuple<int32, int32, int32, double>> *results,
                    KwsLexicographicFst *matched_seq) {
  using namespace fst;
  KwsLexicographicFst keyword_fst;
  KwsLexicographicFst result_fst;
  Map(keyword, &keyword_fst, VectorFstToKwsLexicographicFstMapper());
  Compose(keyword_fst, index, &result_fst);

  if (matched_seq != NULL) {
    *matched_seq = result_fst;
  }

  Project(&result_fst, PROJECT_OUTPUT);
  Minimize(&result_fst, (KwsLexicographicFst *) nullptr, kDelta, true);
  ShortestPath(result_fst, &result_fst, n_best);
  RmEpsilon(&result_fst);

  // No result found
  if (result_fst.Start() == kNoStateId)
    return true;

  // Got something here
  for (ArcIterator<KwsLexicographicFst>
       aiter(result_fst, result_fst.Start()); !aiter.Done(); aiter.Next()) {
    const KwsLexicographicArc &arc = aiter.Value();

    // We're expecting a two-state FST
    if (result_fst.Final(arc.nextstate) != KwsLexicographicWeight::One()) {
      KALDI_WARN << "The result FST does not have the expected structure";
      return false;
    }
    int32 uid = encode_table.Decode(arc.olabel)->olabel;
    int32 tbeg = arc.weight.Value2().Value1().Value();
    int32 tend = arc.weight.Value2().Value2().Value();
    double score = arc.weight.Value1().Value();

    results->push_back(std::make_tuple(uid, tbeg, tend, score));
  }
  return true;
}

struct ActivePath {
  std::vector<KwsLexicographicArc::Label> path;
  KwsLexicographicArc::Weight weight;
  KwsLexicographicArc::Label last;
};

static bool GenerateActivePaths(const KwsLexicographicFst &proxy,
                                std::vector<ActivePath> *paths,
                                KwsLexicographicFst::StateId cur_state,
                                std::vector<KwsLexicographicArc::Label> cur_path,
                                KwsLexicographicWeight cur_weight) {
  for (fst::ArcIterator<KwsLexicographicFst> aiter(proxy, cur_state);
       !aiter.Done(); aiter.Next()) {
    const KwsLexicographicArc &arc = aiter.Value();
    KwsLexicographicWeight temp_weight = Times(arc.weight, cur_weight);

    cur_path.push_back(arc.ilabel);

    if ( arc.olabel != 0 ) {
      ActivePath path;
      path.path = cur_path;
      path.weight = temp_weight;
      path.last = arc.olabel;
      paths->push_back(path);
    } else {
      GenerateActivePaths(proxy, paths,
                          arc.nextstate, cur_path, temp_weight);
    }
    cur_path.pop_back();
  }

  return true;
}

void ComputeDetailedStatistics(
    const KwsLexicographicFst &keyword,
    const fst::internal::EncodeTable<KwsLexicographicArc> &encode_table,
    std::vector<std::tuple<int32, int32, int32, double> > *stats,
    std::vector<std::vector<KwsLexicographicArc::Label> > *ilabels) {
  std::vector<ActivePath> paths;

  if (keyword.Start() == fst::kNoStateId)
    return;

  GenerateActivePaths(keyword, &paths, keyword.Start(),
                      std::vector<KwsLexicographicArc::Label>(),
                      KwsLexicographicWeight::One());

  for (int i = 0; i < paths.size(); ++i) {
    int32 uid = encode_table.Decode(paths[i].last)->olabel;
    int32 tbeg = paths[i].weight.Value2().Value1().Value();
    int32 tend = paths[i].weight.Value2().Value2().Value();
    double score = paths[i].weight.Value1().Value();

    stats->push_back(std::make_tuple(uid, tbeg, tend, score));
    ilabels->push_back(paths[i].path);
  }
}

} // end namespace kaldi
