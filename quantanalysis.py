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
from multiprocessing import Process
import preflibgenprofiles
from math import isnan


methods = ["av", "pav",  "slav", "seqpav", "revseqpav", "seqphrag", "mav",
           "geom1.5", "geom2", "geom5", "monroe", "seqcc",  "cc"]
methods_full = ["AV", "PAV", "SLAV", "seq-PAV", "r.-s.-PAV", "seq-Phr.", "MAV",
                "1.5-Geo.", "2-Geo.", "5-Geo.", "Monroe", "seq-CC", "CC"]


# checks whether a profile is "interesting",
# i.e., winning committees for AV and CC are significantly different
def interestingprofile(inst, discrepancy=0.9):
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
def generate_profiles_preflib(committeesize, logfile,
                              minthreshold, maxthreshold):
    logfile.write("********** PREFLIB **************\n")
    instances = []

    exp_name = "preflib-inst-k="+str(committeesize)
    picklefile = "instances-"+exp_name+".pickle"
    if not os.path.exists(picklefile):
        logfile.write("generating instances " + exp_name + "\n")
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
            for threshold in range(minthreshold, maxthreshold+1):
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
                        logfile.write("\n   " + str(len(instances))
                                      + " instances generated\n\n")
        logfile.write("writing instances to " + picklefile + "\n")
        with open(picklefile, 'w') as f:
            pickle.dump(instances, f)
    else:
        logfile.write("loading instances from " + picklefile + "\n")
        with open(picklefile) as f:
            instances = pickle.load(f)

    return instances


def instance_string(num_inst, num_cand, num_voters,
                    committeesize, setsizes, clones=0):
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
# according to specified distribution
# with uniformly random approval set sizes (chosen from setsizes)
def generate_profiles(model, num_inst, num_cand, num_voters,
                      committeesize, setsizes,
                      minavccdiscrepancy, logfile, clones=0):
    instances = []

    if model == "IC":
        exp_name = "IC," + instance_string(num_inst, num_cand, num_voters,
                                           committeesize, setsizes, clones)
    elif model == "URN":
        exp_name = "URN," + instance_string(num_inst, num_cand, num_voters,
                                            committeesize, setsizes)
    elif model == "MALLOWS":
        exp_name = "MALLOWS," + instance_string(num_inst, num_cand, num_voters,
                                                committeesize, setsizes)
    else:
        logfile.write("model " + str(model) + "unknown\n\n")
        raise Exception

    picklefile = "instances-"+exp_name+".pickle"
    if not os.path.exists(picklefile):
        logfile.write("generating instances\n")
        while len(instances) < num_inst:
            prof = Profile(num_cand * (clones+1))
            appr_sets = []
            if model == "IC":
                for _ in range(num_voters):
                    setsize = random.choice(setsizes)
                    appr = random.sample(range(num_cand), setsize)
                    appr_sets.append([c + i*num_cand for c in appr
                                      for i in range(clones+1)])
            elif model == "URN":
                setsize = random.choice(setsizes)
                replace = random.uniform(0, 1)
                appr_sets = random_urn_profile(num_cand, num_voters,
                                               setsize, replace=replace)
            elif model == "MALLOWS":
                setsize = random.choice(setsizes)
                dispersion = random.uniform(0, 1)
                appr_sets = random_mallows_profile(num_cand, num_voters,
                                                   setsize,
                                                   dispersion=dispersion)

            prof.add_preferences(appr_sets)
            inst = (prof, committeesize)
            if interestingprofile(inst, discrepancy=minavccdiscrepancy):
                instances.append(inst)
                if len(instances) % 25 == 0:
                    logfile.write(str(len(instances))
                                  + " instances generated\n")
        logfile.write("writing instances to " + picklefile + "\n")
        with open(picklefile, 'w') as f:
            pickle.dump(instances, f)
    else:
        logfile.write("loading instances from " + picklefile + "\n")
        with open(picklefile) as f:
            instances = pickle.load(f)

    return instances


# generate Polya Urn profile with fixed size approval sets
def random_urn_profile(num_cand, num_voters, setsize, replace):
    currsize = 1.
    apprsets = []
    replacedsets = {}

    for _ in range(num_voters):
        r = random.random() * currsize
        if r < 1.:
            # base case: sample uniformly at random
            randset = random.sample(range(num_cand), setsize)
            apprsets.append(randset)
            key = tuple(set(randset))
            if key in replacedsets:
                replacedsets[key] += 1
            else:
                replacedsets[key] = 1
            currsize += replace
        else:
            # sample from one of the replaced ballots
            r = random.randint(0, sum(replacedsets.values()))
            for apprset in replacedsets:
                count = replacedsets[apprset]
                if r < count:
                    apprsets.append(list(apprset))
                    break
                else:
                    r -= count

    return apprsets


# generate Mallows profile with fixed size approval sets
def random_mallows_profile(num_cand, num_voters, setsize, dispersion):

    apprsets = []

    reforder = list(range(num_cand))

    prof = preflibgenprofiles.gen_mallows(num_voters, range(num_cand), [1.],
                                          [dispersion], [reforder])

    for ranking, count in prof.items():
        apprsets += [list(ranking[:setsize])] * count

    return apprsets


def plot(rel_avscore, rel_ccscore, exp_name):
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    plt.rcParams.update({'font.size': 14})
    fig, axes = plt.subplots(nrows=2, ncols=1,
                             figsize=(15, 8), sharey=True)

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


# Experiment: variation of AV and CC guarantees of several voting rules
def experiment(mode):
    with open(mode+".txt", "w", buffering=1) as logfile:
        logfile.write("\nExperiment("+mode+")\n")

        minavccdiscrepancy = 0.9
        if mode[-1] == "1":
            setsizes = [2, 3, 4]
        elif mode[-1] == "2":
            setsizes = list(range(5, 11))
        elif mode[-1] == "3":
            setsizes = list(range(11, 21))

        if mode == "preflib":
            a = 4
            b = 20
            instances = []
            for committeesize in range(a, b+1):
                instances += generate_profiles_preflib(committeesize, logfile,
                                                       1, 20)
            exp_name = "preflib-inst-k="+str(a)+"-"+str(b)

        elif mode[:2] == "IC":
            num_inst = 10000
            num_cand = 100
            num_voters = 50
            committeesize = 20
            instances = generate_profiles("IC", num_inst, num_cand, num_voters,
                                          committeesize, setsizes,
                                          minavccdiscrepancy,
                                          logfile)
            logfile.write(str(len(instances)) + " IC instances for k = "
                          + str(committeesize) + "\n")
            exp_name = "IC," + instance_string(
                num_inst, num_cand, num_voters, committeesize, setsizes, 0)

        # with clones
        elif mode == "IC-clones":
            num_inst = 500
            num_cand = 50
            num_voters = 50
            committeesize = 20
            setsizes = [2, 3, 4]
            clones = 3
            instances = generate_profiles("IC", num_inst, num_cand, num_voters,
                                          committeesize, setsizes, clones)
            logfile.write(str(len(instances)) + " IC instances for k = "
                          + str(committeesize) + "\n")
            exp_name = "IC-clones," + instance_string(
                num_inst, num_cand, num_voters, committeesize,
                setsizes, logfile, clones)

        elif mode[:7] == "MALLOWS":
            num_inst = 10000
            num_cand = 100
            num_voters = 50
            committeesize = 20

            instances = generate_profiles("MALLOWS", num_inst, num_cand,
                                          num_voters, committeesize, setsizes,
                                          minavccdiscrepancy, logfile)
            logfile.write(str(len(instances)) + " Mallows instances for k = "
                          + str(committeesize) + "\n")
            exp_name = "MALLOWS," + instance_string(
                num_inst, num_cand, num_voters, committeesize, setsizes, 0)

        elif mode[:3] == "URN":
            num_inst = 10000
            num_cand = 100
            num_voters = 50
            committeesize = 20

            instances = generate_profiles("URN", num_inst, num_cand,
                                          num_voters, committeesize, setsizes,
                                          minavccdiscrepancy, logfile)
            logfile.write(str(len(instances)) + " Polya URN instances for k = "
                          + str(committeesize) + "\n")
            exp_name = "URN," + instance_string(
                num_inst, num_cand, num_voters, committeesize, setsizes, 0)

        else:
            logfile.write("mode " + str(mode) + "unknown\n\n")
            raise Exception

        assert(len(methods) == len(methods_full))

        logfile.write("Running experiments for " + str(len(instances))
                      + " instances\n")

        picklefile = "results-"+exp_name+".pickle"
        averappsetsize = 0
        if not os.path.exists(picklefile):
            rel_avscore = {}
            rel_ccscore = {}
            for method in methods:
                rel_avscore[method] = []
                rel_ccscore[method] = []

            for count, instance in enumerate(instances):
                prof, committeesize = instance
                if (count+1) % 25 == 0:
                    logfile.write(str(count+1) + " instances computed\n")
                averappsetsize += (sum([len(p.approved)
                                        for p in prof.preferences]) * 1.
                                   / len(prof.preferences))
                committees = {}
                for method in methods:
                    if method == "av":
                        committees[method] = abc.compute_av(
                            prof, committeesize, resolute=True)
                    elif method == "mav":
                        committees[method] = abc.compute_minimaxav(
                            prof, committeesize, ilp=True, resolute=True)
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
                    elif method == "revseqpav":
                        committees[method] = abc.compute_revseqpav(
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
                        raise Exception

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

            logfile.write("writing results to " + picklefile + "\n")
            with open(picklefile, 'w') as f:
                pickle.dump((rel_avscore, rel_ccscore, averappsetsize), f)
        else:
            logfile.write("loading results from " + picklefile + "\n")
            with open(picklefile) as f:
                rel_avscore, rel_ccscore, averappsetsize = pickle.load(f)

        averappsetsize = averappsetsize * 1. / len(instances)
        logfile.write("\naverage number of approvals: "
                      + str(averappsetsize) + "\n\n")

        plot(rel_avscore, rel_ccscore, exp_name)

        meth_scores = [(method, np.mean(rel_avscore[method]),
                        np.mean(rel_ccscore[method]))
                       for method in methods]

        logfile.write("\nAV ranking\n")
        logfile.write("----------\n")
        meth_scores.sort(key=itemgetter(1), reverse=True)
        for i in range(len(meth_scores)):
            logfile.write('{0: >2}'.format(str(i+1))+". "
                          + '{0: <12}'.format(meth_scores[i][0])
                          + ": " + "{0:.3f}".format(meth_scores[i][1]) + "\n")

        logfile.write("\nCC ranking\n")
        logfile.write("----------\n")
        meth_scores.sort(key=itemgetter(2), reverse=True)
        for i in range(len(meth_scores)):
            logfile.write('{0: <2}'.format(str(i+1))+". "
                          + '{0: <12}'.format(meth_scores[i][0])
                          + ": " + "{0:.3f}".format(meth_scores[i][2]) + "\n")

        # statistical significance
        logfile.write("\n\n\nStatistical significance:\n")
        for meth1, meth2 in itertools.combinations(methods, 2):
            _, pvalue = stats.ttest_rel(np.asarray(rel_avscore[meth1]),
                                        np.asarray(rel_avscore[meth2]))
            if pvalue > 0.01 or isnan(pvalue):
                logfile.write("relative AV-scores for " + meth1
                              + " and " + meth2)
                logfile.write(" not significant, p = " + str(pvalue) + "\n")

            _, pvalue = stats.ttest_rel(np.asarray(rel_ccscore[meth1]),
                                        np.asarray(rel_ccscore[meth2]))
            if pvalue > 0.01 or isnan(pvalue):
                logfile.write("relative CC-scores for " + meth1
                              + " and " + meth2)
                logfile.write(" not significant, p = " + str(pvalue) + "\n")

        logfile.write("\n\nCompleted.\n")
    print(mode + " done")


#
# run experiments for different preference distribution
#

random.seed(31415)

# run the following experiments in parallel
exps = ["URN1", "MALLOWS1", "preflib", "IC1"]
procs = []
for exp in exps:
    procs.append(Process(target=experiment, args=(exp,)))
    procs[-1].start()

for process in procs:
    process.join()
