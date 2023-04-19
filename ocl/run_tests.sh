# =================
# mini_Imagenet / Online
# =================
python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data mini_imagenet --ns_type noise --ns_factor 0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6 --online True > logs/5runs-mini_imagenet-noise-online.txt

python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data mini_imagenet --ns_type occlusion --ns_factor 0.0 0.07 0.13 0.2 0.27 0.33 0.4 0.47 0.53 0.6 --online True > logs/5runs-mini_imagenet-occlusion-online.txt

python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data mini_imagenet --ns_type blur --ns_factor 0.0 0.28 0.56 0.83 1.11 1.39 1.67 1.94 2.22 2.5 --online True > logs/5runs-mini_imagenet-blur-online.txt

# =================
# mini_Imagenet / Offline
# =================
python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data mini_imagenet --ns_type noise --ns_factor 0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6 --online False > logs/5runs-mini_imagenet-noise-offline.txt

python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data mini_imagenet --ns_type occlusion --ns_factor 0.0 0.07 0.13 0.2 0.27 0.33 0.4 0.47 0.53 0.6 --online False > logs/5runs-mini_imagenet-occlusion-offline.txt

python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data mini_imagenet --ns_type blur --ns_factor 0.0 0.28 0.56 0.83 1.11 1.39 1.67 1.94 2.22 2.5 --online False > logs/5runs-mini_imagenet-blur-offline.txt


# =================
# core50 / Online
# =================
python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data core50 --ns_type noise --ns_factor 0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6 --online True > logs/5runs-core50-noise-online.txt

python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data core50 --ns_type occlusion --ns_factor 0.0 0.07 0.13 0.2 0.27 0.33 0.4 0.47 0.53 0.6 --online True > logs/5runs-core50-occlusion-online.txt

python general_main.py  --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data core50 --ns_type blur --ns_factor 0.0 0.28 0.56 0.83 1.11 1.39 1.67 1.94 2.22 2.5 --online True > logs/5runs-core50-blur-online.txt

# =================
# core50 / Offline
# =================
python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data core50 --ns_type noise --ns_factor 0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6 --online False > logs/5runs-core50-noise-offline.txt

python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data core50 --ns_type occlusion --ns_factor 0.0 0.07 0.13 0.2 0.27 0.33 0.4 0.47 0.53 0.6 --online False > logs/5runs-core50-occlusion-offline.txt

python general_main.py --seed 0 --agent ER --retrieve MIR --update random --error_analysis True --plot_sample False --mem_size 5000 --cl_type ni --num_runs 5 --data core50 --ns_type blur --ns_factor 0.0 0.28 0.56 0.83 1.11 1.39 1.67 1.94 2.22 2.5 --online False > logs/5runs-core50-blur-offline.txt





# --seed 0
# --agent ER
# --retrieve MIR
# --update random
# --error_analysis True
# --plot_sample False
# --mem_size 5000
# --cl_type ni
# --num_runs 5
# --data mini_imagenet
# --ns_type noise
# --ns_factor 0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6
# --online False
# > logs/5runs-mini_imagenet-noise-online.txt




# --ns_type noise
# --ns_factor 0.0 0.4 0.8 1.2 1.6 2.0 2.4 2.8 3.2 3.6 # Noise

# --ns_type occlusion
# --ns_factor 0.0 0.07 0.13 0.2 0.27 0.33 0.4 0.47 0.53 0.6 # Occlusion

# --ns_type blur
# --ns_factor 0.0 0.28 0.56 0.83 1.11 1.39 1.67 1.94 2.22 2.5 # Blur

# --online True # for offline training



# Unused:
# --store True
# --save-path