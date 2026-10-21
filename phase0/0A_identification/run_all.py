"""run_all.py - Chay toan bo kiem toan Pha 0A (G0) theo thu tu, luu log vao thu muc logs/.
Cach dung:  python run_all.py          (day du, uoc tinh 60-90 phut tren 2 nhan (buoc 09, 10, 11 lau nhat))
            python run_all.py quick    (rut gon de kiem tra moi truong, ~1 phut; KHONG dung de ket luan)
Yeu cau: python>=3.9, numpy, scipy.  Dat tat ca file .py cung mot thu muc."""
import subprocess, sys, time, os, pathlib
here=pathlib.Path(__file__).resolve().parent; logs=here/"logs"; logs.mkdir(exist_ok=True)
quick=len(sys.argv)>1 and sys.argv[1]=="quick"
it, B, sc = ("300","200","0.2") if quick else ("2000","500","1.0")
sh = "0.1" if quick else "1.0"   # g0_shared: moi o = 800*sh mau; day du ~10-20 phut moi buoc
STEPS=[
 ("01_power_chan_doan",   ["diag_power.py"], "Chay lai logic power goc + doi chung am (nhieu x10, c2~0)"),
 ("02_g0_doi_chung_am",   ["g0_final.py","neg",it,B], "G0: doi chung am; phai tut ve ~0.33-0.47"),
 ("03_g0_bootstrap_seed_c2_0.047", ["g0_final.py","sweep",it,B,"0.047"], "G0 voi bootstrap theo seed (do phu CI thap: bieu hien cua 3 seed it)"),
 ("04_g0_bootstrap_seed_c2_0.2",   ["g0_final.py","sweep",it,B,"0.2"],   "nhu tren, c2=0.2"),
 ("05_g0_bootstrap_tham_so",       ["g0_pboot.py","0.047",it], "G0 voi bootstrap tham so: do phu & phan loai (be mat parabol ly tuong)"),
 ("06_robust_collapse_plateau",    ["g0_robust2.py"], "3 lop vs 5 lop; naive vs loai diem collapse; vach 5.0 vs 0.5"),
 ("07_robust_buoc_luoi",          ["g0_robust3.py"], "quet buoc luoi 2/sqrt2/1.25 x seed 5,8 duoi collapse+plateau"),
 ("08_robust_mo_hinh_sai",        ["g0_robustness.py",sc], "QUAN TRONG NHAT: hieu chinh tren be mat A, kiem tren be mat B"),
 ("09_estimator_be_mat_chia_se",  ["g0_shared.py",sh], "estimator be mat chia se vs mask tren 9 be mat (lap lai buoc 08)"),
 ("10_chia_se_khi_cong_doi_theo_rank", ["g0_shared_stress.py",sh], "pha gia dinh: do cong thay doi theo rank (gamma)"),
 ("11_giao_thuc_dong_thuan",      ["g0_protocol.py",sh], "GIAO THUC CUOI: phan loai bang shared + co chan dong thuan voi mask; do unsafe% theo gamma"),
]
summary=[]
for name,cmd,desc in STEPS:
    t=time.time(); print(f"\n=== {name}: {desc}",flush=True)
    p=subprocess.run([sys.executable,*cmd],cwd=here,capture_output=True,text=True)
    (logs/f"{name}.txt").write_text(p.stdout+("\n--- STDERR ---\n"+p.stderr if p.stderr.strip() else ""),encoding="utf-8")
    print(p.stdout[-3000:] if p.returncode==0 else "LOI:\n"+p.stderr[-1500:],flush=True)
    summary.append((name,p.returncode,time.time()-t))
print("\n=== TOM TAT ===")
for n,rc,s in summary: print(f"{n:<36} {'OK ' if rc==0 else 'LOI'} {s:6.0f}s")
print(f"Log day du: {logs}")
