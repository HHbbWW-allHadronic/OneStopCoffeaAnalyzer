import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
import warnings
from matplotlib import pyplot as plt
import analyzer
from analyzer.plotting.annotations import addCutTable as act
warnings.filterwarnings("ignore", message=r".*Removed bins.*")
from pathlib import Path
# backgrounds = ["Skim_QCDInclusive2018"]
# compressed = [
#     f"signal_312_{p}" for p in ("1500_1900", "1200_1100", "1500_1400", "2000_1700")
# ]
# uncompressed = [
#     f"signal_312_{p}" for p in ("2000_900", "1200_400", "1500_600", "1500_400")
# ]
# both = compressed + uncompressed

# dense_2000 = [f"signal_312_2000_{p}" for p in (1900, 1700, 1600, 1500, 1300, 1200, 900)]
# dense_1500 = [
#     f"signal_312_1500_{p}" for p in (1450, 1350, 1300, 1200, 1100, 1000, 900, 600, 400)
# ]
# dense_1200 = [f"signal_312_1200_{p}" for p in (1100, 1000, 900, 800, 700, 600, 400)]

# representative = [
#     f"signal_312_{p}"
#     for p in ("2000_1900", "1200_400", "1500_900", "1500_1400", "1200_1100", "2000_900")
# ]

# plotter = Plotter(
#     ['run3_2022D_ak8pf420.pkl','2018dataonly.pkl'],
#     "figurestempdelete",
#     default_backgrounds=None,
#     target_lumi=59.83,
#     coupling="cr",
# )


directory='figures_2018_2022D_test_new_cuts'
sample_names = ["Data2018","Data2022DTemp"]
plotter = Plotter(
    ['2018and2022Danalyzed.pkl'],
    directory,
    default_backgrounds=None,
    target_lumi=59.83,
    coupling="cr",
    non_scaled_histos=True,
    year="2018/2022",
)

def cutsPlot(cut_dict):
    fig, ax = plt.subplots()
    ax.grid(False)
    ax.set_axis_off()
    act(ax,cut_dict,loc="center")
    fig.savefig(Path(directory) / "cuts_table.pdf")
    plt.close(fig)
cutsPlot(plotter.cut_table_dict)  

def cutflowPlot(histogram_name,percent=False):
    fig = plt.figure()
    for name in sample_names:
        h = plotter.non_scaled_histos[histogram_name][name][1:]
        if percent:
            h = h/h[0]
        h.plot1d(label=name)
    if percent:
        plt.ylabel("Normalized Passed Events")
    else:
        plt.ylabel("N Passed Events")
        plt.yscale('log')
    if histogram_name != 'N-1':
        h_labels = plotter.non_scaled_histos_labels[histogram_name][sample_names[0]][1:]
    else:
        h_labels = list()
        for label in plotter.non_scaled_histos_labels[histogram_name][sample_names[0]][1:]:
            h_labels.append(label.replace("N","All"))
    
    if len(plt.gca().get_xticks()) != len(h_labels):
        h_labels = list(dict.fromkeys(h_labels))[:-1]
        plt.xticks(plt.gca().get_xticks(), h_labels, rotation=45)

    plt.legend()
    fig.tight_layout()

    if percent:
        fig.savefig(f'{directory}/{histogram_name}p.pdf')
    else:
        fig.savefig(f'{directory}/{histogram_name}.pdf')
    plt.close(fig)


cutflowPlot('cutflow',True)

cutflowPlot('cutflow')
cutflowPlot('onecut')
cutflowPlot('N-1')

list_of_2d_hists = ['m14_vs_m13', 'ratio_m14_vs_m13', 'm14_vs_m24', 'ratio_m14_vs_m24', 'm14_vs_m12', 'ratio_m14_vs_m12', 
                    'm14_vs_m23', 'ratio_m14_vs_m23', 'm13_vs_m24', 'ratio_m13_vs_m24', 'm13_vs_m12', 'ratio_m13_vs_m12', 'm13_vs_m23', 'ratio_m13_vs_m23', 
                    'm24_vs_m12', 'ratio_m24_vs_m12', 'm24_vs_m23', 'ratio_m24_vs_m23', 'm12_vs_m23', 'ratio_m12_vs_m23','d_phi_01_vs_02', 
                    'd_phi_01_vs_03', 'd_phi_01_vs_12', 'd_phi_01_vs_13', 'd_phi_01_vs_23', 'd_phi_02_vs_03', 'd_phi_02_vs_12', 'd_phi_02_vs_13', 
                    'd_phi_02_vs_23', 'd_phi_03_vs_12', 'd_phi_03_vs_13', 'd_phi_03_vs_23', 'd_phi_12_vs_13', 'd_phi_12_vs_23', 'd_phi_13_vs_23']
list_of_1d_hists = ['h_njet', 'm14_pt', 'm14_eta', 'm14_m', 'm13_pt', 'm13_eta', 'm13_m', 'm24_pt', 'm24_eta', 'm24_m', 'm12_pt', 'm12_eta', 'm12_m', 
                    'm23_pt', 'm23_eta', 'm23_m', 'pt_0', 'eta_0', 'phi_0', 'pt_1', 
                    'eta_1', 'phi_1', 'pt_2', 'eta_2', 'phi_2', 'pt_3', 'eta_3', 'phi_3', 'd_eta_1_1', 'd_phi_1_1', 'd_r_1_1', 'd_eta_1_2', 'd_phi_1_2', 
                    'd_r_1_2', 'd_eta_1_3', 'd_phi_1_3', 'd_r_1_3', 'd_eta_1_4', 'd_phi_1_4', 'd_r_1_4', 'd_eta_2_2', 'd_phi_2_2', 'd_r_2_2', 'd_eta_2_3', 
                    'd_phi_2_3', 'd_r_2_3', 'd_eta_2_4', 'd_phi_2_4', 'd_r_2_4', 'd_eta_3_3', 'd_phi_3_3', 'd_r_3_3', 'd_eta_3_4', 'd_phi_3_4', 'd_r_3_4', 
                    'd_eta_4_4', 'd_phi_4_4', 'd_r_4_4', 'pt_ht_ratio_1', 'pt_ht_ratio_2', 'pt_ht_ratio_3', 'pt_ht_ratio_4', 'pt_ht_ratio_5']
list_of_b_hists = ['loose_bjet_pt', 
                    'loose_nb', 'loose_b_0_pt', 'loose_b_1_pt', 'loose_b_2_pt', 'loose_b_3_pt', 'loose_bb_eta', 'loose_bb_phi', 'loose_bdr', 'medium_bjet_pt', 
                    'medium_nb', 'medium_b_0_pt', 'medium_b_1_pt', 'medium_b_2_pt', 'medium_b_3_pt', 'medium_bb_eta', 'medium_bb_phi', 'medium_bdr', 
                    'medium_b_m', 'medium_b_pt']
list_of_1d_truth_hists = ['truth_stop_pt','truth_stop_eta','truth_stop_phi','truth_stop_b_pt', 'truth_chi_b_pt', 'truth_chi_pt', 'truth_chi_s_pt', 'truth_chi_d_pt', 'truth_stop_b_phi', 
                       'truth_chi_b_phi', 'truth_chi_phi', 'truth_chi_s_phi', 'truth_chi_d_phi','truth_stop_b_eta', 
                       'truth_chi_b_eta', 'truth_chi_eta', 'truth_chi_s_eta', 'truth_chi_d_eta','truth_chi_chi_b_d_eta','truth_chi_chi_b_d_phi', 'truth_chi_chi_b_d_r',
                       'truth_chi_stop_b_d_eta','truth_chi_stop_b_d_phi', 'truth_chi_stop_b_d_r']
list_of_2d_truth_hists = ['truth_chi_pt_v_chi_b_pt', 'truth_chi_pt_v_stop_b_pt', 'truth_chi_eta_v_chi_b_eta',"truth_chi_eta_v_stop_b_eta",'truth_chi_phi_v_chi_b_phi',"truth_chi_phi_v_stop_b_phi"]
list_of_2022d_hists = ['HT', 'h_njet', 'm14_pt', 'm14_eta', 'm14_m', 'm13_pt', 'm13_eta', 'm13_m', 'm24_pt', 
                       'm24_eta', 'm24_m', 'm13_vs_m14', 'ratio_m13_vs_m14', 'm24_vs_m14', 'ratio_m24_vs_m14', 
                       'pt_0', 'eta_0', 'phi_0', 'pt_1', 'eta_1', 'phi_1', 'pt_2', 'eta_2', 'phi_2', 'pt_3', 
                       'eta_3', 'phi_3', 'd_eta_1_2', 'd_phi_1_2', 'd_r_1_2', 'd_eta_1_3', 'd_phi_1_3', 'd_r_1_3', 
                       'd_eta_1_4', 'd_phi_1_4', 'd_r_1_4', 'd_eta_2_3', 'd_phi_2_3', 'd_r_2_3', 'd_eta_2_4', 
                       'd_phi_2_4', 'd_r_2_4', 'd_eta_3_4', 'd_phi_3_4', 'd_r_3_4', 'pt_ht_ratio_0', 'pt_ht_ratio_1', 
                       'pt_ht_ratio_2', 'pt_ht_ratio_3']

# for j in list_of_1d_hists:
#     plotter(j, ["signal_312_1500_400", "signal_312_1500_400_mg"],add_label=j+"\nPythia vs MG (black)",add_name="400",normalize=True,ratio=True)
#     plotter(j, ["signal_312_1500_1100", "signal_312_1500_1100_mg"],add_label=j+"\nPythia vs MG (black)",add_name="1100",normalize=True,ratio=True)
#     plotter(j, ["signal_312_1500_1400", "signal_312_1500_1400_mg"],add_label=j+"\nPythia vs MG (black)",add_name="1400",normalize=True,ratio=True)
# for j in list_of_2d_hists:
#     plotter(j,["signal_312_1500_400","signal_312_1500_400_mg"],add_label=j,add_name="400",normalize=True,ratio=True)
#     plotter(j,["signal_312_1500_1100","signal_312_1500_1100_mg"],add_label=j,add_name="1100",normalize=True,ratio=True)
#     plotter(j,["signal_312_1500_1400","signal_312_1500_1400_mg"],add_label=j,add_name="1400",normalize=True,ratio=True)

print(plotter.histos.keys())
for j in plotter.histos.keys():
    plotter(hist_name=j, sig_set=sample_names, normalize=False, add_label=j, ratio=True, energy='13/13.6 TeV',control_region=True,cut_list_in_plot=True,cut_table_in_plot=False)
    plotter(hist_name=j, sig_set=sample_names, normalize=True, add_label=f'{j}_normalized', add_name="normalized", ratio=True, energy='13/13.6 TeV',control_region=True,cut_list_in_plot=True,cut_table_in_plot=False)

sys.exit()


plotter = Plotter(
    "analyzerout/chargino_reco.pkl",
    "figures",
    default_backgrounds=backgrounds,
    default_axis_opts={"number_jets": Plotter.Split},
)
histos = plotter.histos

# plotter("m14_vs_m13", compressed)
# plotter("m14_vs_m24", uncompressed)
plotter("ratio_m14_vs_m13", compressed)
# plotter("ratio_m14_vs_m3_top_3_no_lead_b", uncompressed)
# plotter("m14_vs_m3_top_3_no_lead_b", uncompressed)
# plotter(
#    "m24_m",
#    uncompressed,
#    normalize=False,
#    scale="log",
#    sig_style="hist",
# )
#
# plotter(
#    "m14_m",
#    representative,
#    normalize=False,
#    add_name="all",
#    scale="log",
#    sig_style="hist",
# )
#
# plotter(
#    "m13_m",
#    compressed,
#    normalize=False,
#    scale="log",
#    sig_style="hist",
# )
#
# plotter(
#    "m3_top_3_no_lead_b",
#    uncompressed,
#    normalize=False,
#    scale="log",
#    sig_style="hist",
# )


to_plot = [
    ("max_chi_child_dr", compressed, "compressed"),
    ("max_chi_child_dr", representative, "representative"),
    ("max_chi_child_dr", uncompressed, "uncompressed"),
    ("mean_chi_child_dr", compressed, "compressed"),
    ("mean_chi_child_dr", representative, "representative"),
    ("mean_chi_child_dr", uncompressed, "uncompressed"),
    ("min_chi_child_dr", compressed, "compressed"),
    ("min_chi_child_dr", representative, "representative"),
    ("min_chi_child_dr", uncompressed, "uncompressed"),
    ("max_chi_child_dr", dense_2000, "dense_2000"),
    ("mean_chi_child_dr", dense_2000, "dense_2000"),
    ("min_chi_child_dr", dense_2000, "dense_2000"),
    ("max_chi_child_dr", dense_1500, "dense_1500"),
    ("mean_chi_child_dr", dense_1500, "dense_1500"),
    ("min_chi_child_dr", dense_1500, "dense_1500"),
    ("max_chi_child_dr", dense_1200, "dense_1200"),
    ("mean_chi_child_dr", dense_1200, "dense_1200"),
    ("min_chi_child_dr", dense_1200, "dense_1200"),
    ("max_chi_child_eta", dense_2000, "dense_2000"),
    ("mean_chi_child_eta", dense_2000, "dense_2000"),
    ("min_chi_child_eta", dense_2000, "dense_2000"),
    ("max_chi_child_eta", dense_1500, "dense_1500"),
    ("mean_chi_child_eta", dense_1500, "dense_1500"),
    ("min_chi_child_eta", dense_1500, "dense_1500"),
    ("max_chi_child_eta", dense_1200, "dense_1200"),
    ("mean_chi_child_eta", dense_1200, "dense_1200"),
    ("min_chi_child_eta", dense_1200, "dense_1200"),
    ("max_chi_child_phi", dense_2000, "dense_2000"),
    ("mean_chi_child_phi", dense_2000, "dense_2000"),
    ("min_chi_child_phi", dense_2000, "dense_2000"),
    ("max_chi_child_phi", dense_1500, "dense_1500"),
    ("mean_chi_child_phi", dense_1500, "dense_1500"),
    ("min_chi_child_phi", dense_1500, "dense_1500"),
    ("max_chi_child_phi", dense_1200, "dense_1200"),
    ("mean_chi_child_phi", dense_1200, "dense_1200"),
    ("min_chi_child_phi", dense_1200, "dense_1200"),
    ("max_chi_child_dr_all_three", compressed, "compressed"),
    ("max_chi_child_dr_all_three", representative, "representative"),
    ("max_chi_child_dr_all_three", uncompressed, "uncompressed"),
    ("mean_chi_child_dr_all_three", compressed, "compressed"),
    ("mean_chi_child_dr_all_three", representative, "representative"),
    ("mean_chi_child_dr_all_three", uncompressed, "uncompressed"),
    ("min_chi_child_dr_all_three", compressed, "compressed"),
    ("min_chi_child_dr_all_three", representative, "representative"),
    ("min_chi_child_dr_all_three", uncompressed, "uncompressed"),
    ("max_chi_child_eta", compressed, "compressed"),
    ("max_chi_child_eta", representative, "representative"),
    ("max_chi_child_eta", uncompressed, "uncompressed"),
    ("mean_chi_child_eta", compressed, "compressed"),
    ("mean_chi_child_eta", representative, "representative"),
    ("mean_chi_child_eta", uncompressed, "uncompressed"),
    ("min_chi_child_eta", compressed, "compressed"),
    ("min_chi_child_eta", representative, "representative"),
    ("min_chi_child_eta", uncompressed, "uncompressed"),
    ("max_chi_child_phi", compressed, "compressed"),
    ("max_chi_child_phi", representative, "representative"),
    ("max_chi_child_phi", uncompressed, "uncompressed"),
    ("mean_chi_child_phi", compressed, "compressed"),
    ("mean_chi_child_phi", representative, "representative"),
    ("mean_chi_child_phi", uncompressed, "uncompressed"),
    ("min_chi_child_phi", compressed, "compressed"),
    ("min_chi_child_phi", representative, "representative"),
    ("min_chi_child_phi", uncompressed, "uncompressed"),
    ("stop_pt", representative, "representative"),
]

for n, s, a in to_plot:
    plotter(
        n,
        s,
        [],
        add_name=a,
        normalize=True,
        scale="linear",
        sig_style="hist",
    )
