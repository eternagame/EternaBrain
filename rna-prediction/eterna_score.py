'''
Function that will take RNA sequence as input and output Eternabot score for reward for reinforcement new_element
@authot: Rohan Koodli
'''

import sys
import numpy as np
import RNA
import ensemble_utils
from eterna_utils import get_dotplot, get_rna_elements_from_secstruct, get_pairmap_from_secstruct
def find_parens(s):
    toret = {}
    pstack = []

    for i, c in enumerate(s):
        if c == '(':
            pstack.append(i)
        elif c == ')':
            if len(pstack) == 0:
                raise IndexError("No matching closing parens at: " + str(i))
            toret[pstack.pop()] = i

    if len(pstack) > 0:
        raise IndexError("No matching opening parens at: " + str(pstack.pop()))

    return toret

def convert(base_seq):
    A = np.array([1.,0.,0.,0.])
    U = np.array([0.,1.,0.,0.])
    G = np.array([0.,0.,1.,0.])
    C = np.array([0.,0.,0.,1.])
    str_struc = []
    for i in base_seq:
        if np.array_equal(i,A):
            str_struc.append('A')
        elif np.array_equal(i,U):
            str_struc.append('U')
        elif np.array_equal(i,G):
            str_struc.append('G')
        elif np.array_equal(i,C):
            str_struc.append('C')
    struc = ''.join(str_struc)
    s,e = RNA.fold(struc)
    return s,e,struc

def eternabot_score(seq):
    struc,fe,str_seq = convert(seq)
    attrs_dict = {'secstruct':struc, 'sequence':str_seq, 'fe':fe}
    ua_counter = 0
    gu_counter = 0
    gc_counter = 0
    for key,val in find_parens(struc).iteritems():
        general = []
        l = seq[key] + seq[val]
        if l == 'GC' or l == 'CG':
            gc_counter += 1
        elif l == 'GU' or l == 'UG':
            gu_counter += 1
        elif l == 'AU' or l == 'UA':
            ua_counter += 1

    attrs_dict['ua'] = ua_counter
    attrs_dict['gu'] = gu_counter
    attrs_dict['gc'] = gc_counter
    total_pairs = ua_counter + gu_counter + gc_counter

    attrs_dict['pairmap'] = get_pairmap_from_secstruct(struc)
    attrs_dict['dotplot'] = get_dotplot(struc)
    attrs_dict['secstruct_elements'] = get_rna_elements_from_secstruct(struc)
    attrs_dict['meltpoint'] = 97.0

    strategy_names = ["merryskies_only_as_in_the_loops", "aldo_repetition", "dejerpha_basic_test", "eli_blue_line", "clollin_gs_in_place", "quasispecies_test_by_region_boundaries", "eli_gc_pairs_in_junction", "eli_no_blue_nucleotides_in_hook", "mat747_31_loops", "merryskies_1_1_loop", "xmbrst_clear_plot_stack_caps_and_safe_gc", "jerryp70_jp_stratmark", "eli_energy_limit_in_tetraloops", "eli_double_AUPair_strategy", "eli_green_blue_strong_middle_half", "eli_loop_pattern_for_small_multiloops", "eli_tetraloop_similarity", "example_gc60", "penguian_clean_dotplot", "eli_twisted_basepairs", "aldo_loops_and_stacks", "eli_direction_of_gc_pairs_in_multiloops_neckarea", "eli_multiloop_similarity", "eli_green_line", "ding_quad_energy", "quasispecies_test_by_region_loops", "berex_berex_loop_basic", "eli_legal_placement_of_GUpairs", "merryskies_1_1_loop_energy", "ding_tetraloop_pattern", "aldo_mismatch", "eli_tetraloop_blues", "eli_red_line", "eli_wrong_direction_of_gc_pairs_in_multiloops", "deivad_deivad_strategy", "eli_direction_of_gc_pairs_in_multiloops", "eli_no_blue_nucleotides_strategy", "berex_basic_test", "eli_numbers_of_yellow_nucleotides_pr_length_of_string", "kkohli_test_by_kkohli"]
    ensemble = ensemble_utils.Ensemble("L2", strategy_names, None)
    # try:
    #     scores = ensemble.score(attrs_dict)
    # except AttributeError:
    #     return 0
    if total_pairs == 0:
        return 0
    else:
        scores = ensemble.score(attrs_dict)
        return scores


# seq = "GGAAAUCC"
# struc = '(((..)))'
# print find_parens(struc)
# print find_parens(struc).keys()
# print type(find_parens(struc).values())
# ua_counter = 0
# gu_counter = 0
# gc_counter = 0
# for key,val in find_parens(struc).iteritems():
#     general = []
#     print seq[key],seq[val]
#     l = seq[key] + seq[val]
#     if l == 'GC' or l == 'CG':
#         gc_counter += 1
#     elif l == 'GU' or l == 'UG':
#         gu_counter += 1
#     elif l == 'AU' or l == 'UA':
#         ua_counter += 1
#
# print ua_counter,gu_counter,gc_counter
