import abcvoting.rules_approval as abc
from abcvoting.preferences import Profile
import matplotlib.pyplot as plt
import random
import numpy as np
from operator import itemgetter
import preflibio
import pickle
import os.path
import abcvoting.score_functions as sf
from gmpy2 import mpq
from scipy import stats
import itertools


plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


# checks whether a profile is "interesting",
# i.e., winning committees for AV and CC are significantly different
def interestingprofile(inst, discrepancy=0.85):
    prof, committeesize = inst

    # enough candidates to fill committee
    appr = set()
    for pref in prof.preferences:
        appr.update(pref.approved)
    if len(appr) < committeesize:
        return False

    # voters must not approve more than 90% of candidates on average
    aver = np.mean([len(pref.approved) for pref in prof.preferences])
    if aver > 0.9*len(appr):
        return False

    # meaningful discrepancy between AV and CC
    committeesav = abc.compute_av(prof, committeesize, resolute=True)
    committeescc = abc.compute_cc(prof, committeesize,
                                  ilp=True, resolute=True)

    avav = [sf.thiele_score(prof, comm, 'av') for comm in committeesav]
    avcc = [sf.thiele_score(prof, comm, 'av') for comm in committeescc]
    assert min(avav) == max(avav)
    cccc = [sf.thiele_score(prof, comm, 'cc') for comm in committeescc]
    ccav = [sf.thiele_score(prof, comm, 'cc') for comm in committeesav]
    assert min(cccc) == max(cccc)
    if max(ccav) <= discrepancy * max(cccc) and \
       max(avcc) <= discrepancy * max(avav):
        return True
    else:
        return False


# converts a ranking-based profile into an approval profile
# input as obtained from preflib data
def approvalsetsfromrankings(rankmap, candmap, threshold=1):
    # threshold = c means that only candidates
    # in the first c levels are approved
    approved = []
    for candidate, pos in rankmap.iteritems():
        if pos <= threshold:
            if candidate < min(candmap) or candidate > max(candmap):
                print ("candidate", candidate,
                       "not valid (should be in the interval between",
                       min(candmap), "and", max(candmap), ")")
                raise(BaseException)
            approved.append(candidate-min(candmap))
    return approved


# reads preflib data and generates suitable test data
def generate_profiles_preflib(committeesize):
    print "********** PREFLIB **************"
    instances = []

    exp_name = "preflib-inst-k="+str(committeesize)
    picklefile = "instances-"+exp_name+".pickle"
    if not os.path.exists(picklefile):
        print "generating instances", exp_name
        files = os.listdir("preflibdata/.")
        for fn in files:
            inf = open("preflibdata/"+fn, "r")
            candmap, rankmaps, rankmapcounts, num_votes =\
                preflibio.read_election_file(inf)
            num_cand = max(candmap)-min(candmap)+1
            if num_cand > 100:
                # too many candidates
                continue
            elif num_cand < committeesize+2:
                # too few candidates
                continue
            elif num_votes > 2000:
                # too many votes
                continue
            for threshold in range(1, committeesize):
                # the *threshold* many top levels of weak orders are approved
                prof = Profile(num_cand)
                prefs = [list(approvalsetsfromrankings(rankmaps[i],
                                                       candmap, threshold))
                         for i in range(len(rankmaps))
                         for _ in range(rankmapcounts[i])]
                prof.add_preferences(prefs)
                inst = (prof, committeesize)
                if interestingprofile(inst, discrepancy=0.9):
                    instances.append(inst)
                    if len(instances) % 25 == 0:
                        print ""
                        print "   ", len(instances), "instances generated"
                        print ""
        print "writing instances to", picklefile
        with open(picklefile, 'w') as f:
            pickle.dump(instances, f)
    else:
        print "loading instances from", picklefile
        with open(picklefile) as f:
            instances = pickle.load(f)

    return instances


def ic_instance_string(num_inst, num_cand, num_voters,
                       committeesize, setsizes, clones):
    if len(setsizes) == 1:
        exp_name = ("inst=" + str(num_inst) + ",m=" + str(num_cand) + ",n="
                    + str(num_voters) + ",k=" + str(committeesize)
                    + ",setsize=" + str(setsizes[0])
                    + ",clones=" + str(clones))
    else:
        exp_name = ("inst=" + str(num_inst) + ",m=" + str(num_cand) + ",n="
                    + str(num_voters) + ",k="+str(committeesize)
                    + ",setsizes=" + str(setsizes)
                    + ",clones=" + str(clones))
    return exp_name


# generates suitable test data randomly
# according to the Independent Culture model
# with random approval set sizes (chosen from setsizes)
def generate_profiles_IC(num_inst, num_cand, num_voters,
                         committeesize, setsizes, clones=0):
    instances = []

    exp_name = "IC," + ic_instance_string(num_inst, num_cand, num_voters,
                                          committeesize, setsizes, clones)
    picklefile = "instances-"+exp_name+".pickle"
    if not os.path.exists(picklefile):
        print "generating instances"
        while len(instances) < num_inst:
            prof = Profile(num_cand * (clones+1))
            for _ in range(num_voters):
                setsize = random.choice(setsizes)
                appr_set = (random.sample(range(num_cand), setsize))
                appr_set += [c + i*num_cand for c in appr_set
                             for i in range(clones+1)]
                prof.add_preferences(appr_set)
            inst = (prof, committeesize)
            if interestingprofile(inst):
                instances.append(inst)
                if len(instances) % 25 == 0:
                    print len(instances), "instances generated"
        print "writing instances to", picklefile
        with open(picklefile, 'w') as f:
            pickle.dump(instances, f)
    else:
        print "loading instances from", picklefile
        with open(picklefile) as f:
            instances = pickle.load(f)

    return instances


# Experiment 1: variation of AV and CC scores of several voting rules
def exp1(mode):

    print ""
    print "Running exp1("+mode+")"
    print ""

    if mode == "preflib":
        a = 4
        b = 20
        instances = []
        for committeesize in range(a, b+1):
            instances += generate_profiles_preflib(committeesize)

        exp_name = "preflib-inst-k="+str(a)+"-"+str(b)

    elif mode == "IC1":
        num_inst = 250
        num_cand = 20
        num_voters = 50
        committeesize = 5
        setsize = 4
        instances = generate_profiles_IC(num_inst, num_cand,
                                         num_voters, committeesize, [setsize])
        print len(instances), "IC instances for k =", committeesize

        exp_name = "IC1," + ic_instance_string(
            num_inst, num_cand, num_voters, committeesize, setsize, 0)

    elif mode == "IC2":
        num_inst = 250
        num_cand = 20
        num_voters = 50
        committeesize = 7
        setsize = 4
        instances = generate_profiles_IC(num_inst, num_cand, num_voters,
                                         committeesize, [setsize])
        print len(instances), "IC instances for k =", committeesize

        exp_name = "IC2," + ic_instance_string(
            num_inst, num_cand, num_voters, committeesize, setsize, 0)

    elif mode == "IC3":
        num_inst = 250
        num_cand = 20
        num_voters = 50
        committeesize = 5
        setsizes = [2, 3, 4, 5]
        instances = generate_profiles_IC(num_inst, num_cand, num_voters,
                                         committeesize, setsizes)
        print len(instances), "IC instances for k =", committeesize

        exp_name = "IC3," + ic_instance_string(
            num_inst, num_cand, num_voters, committeesize, setsizes, 0)

    elif mode == "IC4":
        num_inst = 500
        num_cand = 20
        num_voters = 50
        committeesize = 5
        setsizes = [2, 3, 4, 5]
        instances = generate_profiles_IC(num_inst, num_cand, num_voters,
                                         committeesize, setsizes)
        print len(instances), "IC instances for k =", committeesize

        exp_name = "IC4," + ic_instance_string(
            num_inst, num_cand, num_voters, committeesize, setsizes, 0)

    elif mode == "IC":
        num_inst = 10000
        num_cand = 100
        num_voters = 50
        committeesize = 20
        setsizes = [2, 3, 4]
        clones = 0
        instances = generate_profiles_IC(num_inst, num_cand, num_voters,
                                         committeesize, setsizes)
        print len(instances), "IC instances for k =", committeesize

        exp_name = "IC," + ic_instance_string(
            num_inst, num_cand, num_voters, committeesize, setsizes, 0)

    elif mode == "IC+":
        num_inst = 500
        num_cand = 50
        num_voters = 50
        committeesize = 20
        setsizes = [2, 3, 4]
        clones = 3
        instances = generate_profiles_IC(num_inst, num_cand, num_voters,
                                         committeesize, setsizes, clones)
        print len(instances), "IC instances for k =", committeesize

        exp_name = "IC+," + ic_instance_string(
            num_inst, num_cand, num_voters, committeesize, setsizes, clones)

    else:
        print "mode", mode, "unknown"
        return

    methods = ["av", "pav",  "slav", "seqpav", "seqphrag",
               # "optphrag",
               "geom1.5", "geom2", "geom5", "monroe", "seqcc",  "cc"]
    methods_full = ["AV", "PAV", "SLAV", "seq-PAV", "seq-Phr.",
                    # "opt-Phragmen",
                    "1.5-Geom.", "2-Geom.",
                    "5-Geom.", "Monroe", "seq-CC", "CC"]

    print "Running experiments for", len(instances), "instances"

    picklefile = "results-"+exp_name+".pickle"
    if not os.path.exists(picklefile):
        rel_avscore = {}
        rel_ccscore = {}
        for method in methods:
            rel_avscore[method] = []
            rel_ccscore[method] = []

        weakdominationav = {}
        weakdominationcc = {}
        for m1 in methods:
            for m2 in methods:
                weakdominationav[(m1, m2)] = True
                weakdominationcc[(m1, m2)] = True

        for prof, committeesize in instances:
            committees = {}
            for method in methods:
                if method == "av":
                    committees[method] = abc.compute_av(
                        prof, committeesize, resolute=True)
                elif method == "cc":
                    committees[method] = abc.compute_cc(
                        prof, committeesize, ilp=True, resolute=True)
                elif method == "pav":
                    committees[method] = abc.compute_pav(
                        prof, committeesize, ilp=True, resolute=True)
                elif method == "slav":
                    committees[method] = abc.compute_slav(
                        prof, committeesize, ilp=True, resolute=True)
                elif method == "seqpav":
                    committees[method] = abc.compute_seqpav(
                        prof, committeesize, resolute=True)
                elif method == "seqcc":
                    committees[method] = abc.compute_seqcc(
                        prof, committeesize, resolute=True)
                elif method == "seqphrag":
                    committees[method] = abc.compute_seqphragmen(
                        prof, committeesize, resolute=True)
                elif method == "optphrag":
                    committees[method] = abc.compute_optphragmen_ilp(
                        prof, committeesize, resolute=True)
                elif method == "monroe":
                    committees[method] = abc.compute_monroe(
                        prof, committeesize, ilp=True, resolute=True)
                elif method[:4] == "geom":
                    committees[method] = \
                        abc.compute_thiele_methods_ilp(
                            prof, committeesize, method, resolute=True)
                elif method[:7] == "seqgeom":
                    committees[method] = abc.compute_seq_thiele_methods(
                        prof, committeesize, method[3:], resolute=True)
                else:
                    assert False

            avscore = {}
            ccscore = {}
            for method in methods:
                avscore[method] = sf.thiele_score(
                    prof, committees[method][0], 'av')
                ccscore[method] = sf.thiele_score(
                    prof, committees[method][0], 'cc')

            for method in methods:
                rel_avscore[method].append(
                    float(mpq(avscore[method], avscore["av"])))
                rel_ccscore[method].append(
                    float(mpq(ccscore[method], ccscore["cc"])))

            if rel_avscore["cc"][-1] > 0.9:
                print prof
                print rel_avscore["cc"][-1]
                print abc.compute_cc(prof, committeesize,
                                     ilp=True, resolute=True)
                print abc.compute_av(prof, committeesize)

            for m1 in methods:
                for m2 in methods:
                    if (avscore[m1] < avscore[m2]):
                        weakdominationav[(m1, m2)] = False
                    if (ccscore[m1] < ccscore[m2]):
                        weakdominationcc[(m1, m2)] = False

        if False:
            for m1 in methods:
                for m2 in methods:
                    if m1 == m2:
                        continue
                    if m1 == "av":
                        continue
                    if weakdominationav[(m1, m2)]:
                        print m1, "weakly AV-dominates", m2

            for m1 in methods:
                for m2 in methods:
                    if m1 == m2:
                        continue
                    if m1 == "cc":
                        continue
                    if weakdominationcc[(m1, m2)]:
                        print m1, "weakly CC-dominates", m2

        print "writing results to", picklefile
        with open(picklefile, 'w') as f:
            pickle.dump((rel_avscore, rel_ccscore), f)
    else:
        print "loading results from", picklefile
        with open(picklefile) as f:
            rel_avscore, rel_ccscore = pickle.load(f)

    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8), sharey=True)

    plt.ylim(0.69, 1.01)
    data_to_plot = [rel_avscore[method] for method in methods]
    axes[0].boxplot(data_to_plot, widths=0.8,
                    labels=[""]*len(methods), whis='range')
    data_to_plot = [rel_ccscore[method] for method in methods]
    axes[1].boxplot(data_to_plot, widths=0.8,
                    labels=methods_full, whis='range')
    fig.subplots_adjust(hspace=0.05)
    fig.savefig(str('comparing-scores-'+exp_name+'.pdf').replace(" ", ""),
                bbox_inches='tight')
    plt.close()

    scores = [(method, np.mean(rel_avscore[method]),
               np.mean(rel_ccscore[method]))
              for method in methods]
    print
    print "AV ranking"
    print "----------"
    scores.sort(key=itemgetter(1), reverse=True)
    for i in range(len(scores)):
        print ('{0: >2}'.format(str(i+1))+". "+'{0: <12}'.format(scores[i][0])
               + ": " + "{0:.3f}".format(scores[i][1]))
    print
    print "CC ranking"
    print "----------"
    scores.sort(key=itemgetter(2), reverse=True)
    for i in range(len(scores)):
        print ('{0: <2}'.format(str(i+1))+". "+'{0: <12}'.format(scores[i][0])
               + ": " + "{0:.3f}".format(scores[i][2]))

    # statistical significance
    for meth1, meth2 in itertools.combinations(methods, 2):
        _, pvalue = stats.ttest_rel(np.asarray(rel_avscore[meth1]),
                                    np.asarray(rel_avscore[meth2]))
        if pvalue > 0.01:
            print "relative AV-scores for", meth1, "and", meth2,
            print "not significant, p =", pvalue

        _, pvalue = stats.ttest_rel(np.asarray(rel_ccscore[meth1]),
                                    np.asarray(rel_ccscore[meth2]))
        if pvalue > 0.01:
            print "relative CC-scores for", meth1, "and", meth2,
            print "not significant, p =", pvalue


# run the following experiments
random.seed(31415)
exp1("preflib")
exp1("IC")
