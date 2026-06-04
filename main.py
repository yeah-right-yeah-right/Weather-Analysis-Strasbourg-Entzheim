import csv
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
from datetime import timedelta

os.makedirs("Evolutions", exist_ok=True)
os.makedirs("Spectres", exist_ok=True)
os.makedirs("Mod_an", exist_ok=True)
os.makedirs("Bruits", exist_ok=True)
os.makedirs("Cycles", exist_ok=True)
os.makedirs("Tendances", exist_ok=True)





fields = []
rows = []    

def extract_values(rows, var):
    L=[]
    for i in rows:
        if i[var] is not None and i[var] != '':
            L.append(float(i[var]))
    return L

def stats(values):
    med = np.median(values)
    moy = np.mean(values)
    ec = np.std(values)
    mn = min(values)
    mx = max(values)
    return {
        "mediane" : med ,
        "moyenne" : moy,
        "ecart" : ec,
        "min" : mn,
        "max" : mx
    }

with open('strasbourg_entzheim.csv', newline ='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter = ',')
    for row in reader:     
        row['time'] = datetime.strptime(row['time'], "%Y-%m-%d")
        rows.append(row)
    print("Total no. of rows: %d" % reader.line_num)

rows.sort(key=lambda x: x['time'])

analyze = ['tavg', 'tmin', 'tmax', 'prcp', 'pres', 'wspd', 'tsun']

all_stats = {}
for var in analyze:
    curr = extract_values(rows, var)
    all_stats[var] = stats(curr)

stat_names = all_stats[analyze[0]].keys()

print(f"{'stat':<10}", end="")
for var in analyze:
    print(f"{var:<10}", end="")
print()

for stat in stat_names:
    print(f"{stat:<10}", end="")
    for var in analyze:
        value = all_stats[var][stat]
        if stat == "moyenne" or stat == "ecart":
            value = round(value, 2)
        print(f"{value:<10}", end="")
    print()


depart = rows[0]['time']

def setup_x_y(rows, var):
    x=[]
    y=[]
    for i in rows:
        if i[var] is not None and i[var] != '':
            x.append((i['time']-depart).days)
            y.append(float(i[var]))
    return np.array(x), np.array(y)

def extract_snow(rows):
    dates = []
    snow = []
    for r in rows:
        if r['snow'] not in (None, ''):
            dates.append(r['time'])
            snow.append(float(r['snow']))
    return np.array(dates), np.array(snow)

def construire_fft_pascont(rows, var, depart, max_gap_days=7, min_segment_days=365*3):
    day_to_val = {}
    min_dt = None
    max_dt = None

    for r in rows:
        dt = r["time"]
        if min_dt is None or dt < min_dt:
            min_dt = dt
        if max_dt is None or dt > max_dt:
            max_dt = dt

        v = r.get(var, None)
        if v is not None and v != "":
            try:
                day_to_val[dt.date()] = float(v)
            except ValueError:
                pass
    n_days = (max_dt.date() - min_dt.date()).days + 1
    x_full = np.arange(n_days, dtype=int) 
    y_full = np.full(n_days, np.nan, dtype=float)

    # remplir ce qu'on a
    for i in range(n_days):
        d = (min_dt.date() + timedelta(days=i))  # date python
        if d in day_to_val:
            y_full[i] = day_to_val[d]

    valid = ~np.isnan(y_full)
    if valid.sum() < 10:
        return None, None 

    xi = np.arange(n_days)
    y_interp = y_full.copy()
    y_interp[np.isnan(y_interp)] = np.interp(xi[np.isnan(y_interp)], xi[valid], y_full[valid])

    y_masked = y_interp.copy()
    isn = np.isnan(y_full)
    if isn.any():
        start = None
        for i in range(n_days + 1):
            if i < n_days and isn[i] and start is None:
                start = i
            if start is not None and (i == n_days or not isn[i]):
                run_len = i - start
                if run_len > max_gap_days:
                    y_masked[start:i] = np.nan
                start = None

    good = ~np.isnan(y_masked)
    best_len = 0
    best_a = None
    best_b = None
    a = None

    for i in range(n_days + 1):
        if i < n_days and good[i] and a is None:
            a = i
        if a is not None and (i == n_days or not good[i]):
            b = i
            seg_len = b - a
            if seg_len > best_len:
                best_len = seg_len
                best_a, best_b = a, b
            a = None

    if best_len < min_segment_days:
        if best_len < 200:
            return None, None

    offset = (min_dt - depart).days
    x_seg = (x_full[best_a:best_b] + offset).astype(int)
    y_seg = y_masked[best_a:best_b].astype(float)

    return x_seg, y_seg


def normaliser_poly(x, y, deg=2):
    p = np.polyfit(x, y, deg)
    trend = np.polyval(p, x)
    return y - trend, trend


x_y = dict()

for i in analyze:
    x_y[i]=setup_x_y(rows,i)

pres_xy = construire_fft_pascont(rows, "pres", depart, max_gap_days=7, min_segment_days=365*3)
x_y["pres_fft"] =  pres_xy

x_y['snow'] = extract_snow(rows)

def evolution(var,x_y):
    n=0
    valid = ~np.isnan(x_y[var][1])
    plt.figure(figsize=(10,4))
    plt.plot(x_y[var][0][valid], x_y[var][1][valid])
    plt.xlabel("Temps")
    plt.ylabel(var)
    plt.title("Évolution de "+ var +" à Strasbourg-Entzheim (1950-2024)")
    plt.grid()
    plt.savefig('Evolutions/' +var+'_evolution.png', dpi=150)
    plt.show()


for i in analyze:
    evolution(i,x_y)

def journees_neige(dates, snow):
    years = np.array([d.year for d in dates])
    uq= np.unique(years)

    sd = []

    for y in uq:
        mask = (years == y)
        sd.append(np.sum(snow[mask] > 0))

    return uq, np.array(sd)

snow_x, snow_y = journees_neige(x_y['snow'][0], x_y['snow'][1])

plt.figure(figsize=(9,4))
plt.plot(snow_x, snow_y, 'o-', alpha=0.7)
plt.xlabel("Annee")
plt.ylabel("Nombre des jours snow>0")
plt.title("Évolution de # des journees snow>0 à Strasbourg-Entzheim (1950-2024)")
plt.grid()
plt.savefig("Evolutions/snow_evolution.png", dpi=150)
plt.show()



periode = 365.25

def modele_annuel(a0, a1, b1, x, per):
    return a0 + a1*np.cos(2*np.pi * x/per) + b1*np.sin(2*np.pi * x/per)

def a1(x, y, per):
    return (2/len(y)) * np.sum(y * np.cos(2*np.pi * x/per))

def b1(x, y, per):
    return (2/len(y)) * np.sum(y * np.sin(2*np.pi * x/per))

a_0 = dict()
a_1 = dict()
b_1 = dict()
amplitude = dict()

for i in analyze:
    a_0[i] = np.mean(x_y[i][1])
    a_1[i] = a1(x_y[i][0], x_y[i][1], periode)
    b_1[i] = b1(x_y[i][0], x_y[i][1], periode)
    amplitude[i] = np.sqrt(a_1[i]**2 + b_1[i]**2)

def mod_ann(x_y, a_0, a_1, b_1, amplitude, var):

    amplitudes_fft, periodes_fft, N, y_modele, mask = None, None, None, None, None

    if var in ['tavg','tmin', 'tmax', 'tsun']:

        print("\n--- Modele annuel pour " + var + "--")
        print("X(t) = %.2f + %.2f*cos + %.2f*sin" % (a_0[var], a_1[var], b_1[var]))
        print("Amplitude: %.2f C" % amplitude[var])

        signal = np.array(x_y[var][1])
        signal = signal - np.mean(signal)
        N = len(signal)
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(N, d=1)
        mask = freqs > 0
        amplitudes_fft = np.abs(fft_vals[mask]) * 2 / N
        periodes_fft = 1 / freqs[mask]
        if np.all(np.diff(x_y[var][0]) == np.diff(x_y[var][0])[0]):

            plt.figure(figsize=(8,4))
            plt.plot(freqs[mask], np.abs(fft_vals[mask]))
            plt.xlabel("Fréquence (1/jour)")
            plt.ylabel("Amplitude")
            plt.title("Spectre de Fourier de "+var)
            plt.grid()
            plt.savefig('Spectres/' + var + '_spectre.png', dpi=150)
            plt.show()

            
        else:
            amplitudes_fft, periodes_fft, N = None, None, None
        
        y_modele = modele_annuel(a_0[var], a_1[var], b_1[var], x_y[var][0], periode)
        plt.figure(figsize=(10,4))
        plt.plot(x_y[var][0][:3*365], x_y[var][1][:3*365], 'b-', linewidth=0.5, alpha=0.7, label='Données')
        plt.plot(x_y[var][0][:3*365], y_modele[:3*365], 'r-', linewidth=2, label='Modèle')
        plt.xlabel("Jours")
        plt.ylabel(var)
        plt.title("Modele vs donnees sur 3 ans pour " + var)
        plt.legend()
        plt.grid()
        plt.savefig('Mod_an/' + var + '_modele.png', dpi=150)
        plt.show()

    return amplitudes_fft, periodes_fft, N, y_modele, mask

amplitudes_fft = dict()
periodes_fft = dict()
N = dict()
y_modele = dict()
mask = dict()

for i in analyze:
    fst,snd,trd,frth, ffth = mod_ann(x_y, a_0, a_1, b_1, amplitude, i)
    amplitudes_fft[i]=fst
    periodes_fft[i]=snd
    N[i]=trd
    y_modele[i] = frth
    mask[i] = ffth
#etude pres

if "pres_fft" in x_y:
    x_pres, y_pres = x_y["pres_fft"]

    # detrend + centrage
    y_dt, _ = normaliser_poly(x_pres, y_pres, deg=2)
    y_dt = y_dt - np.mean(y_dt)

    N["pres"] = len(y_dt)
    fft_vals = np.fft.fft(y_dt)
    freqs = np.fft.fftfreq(N["pres"], d=1.0)  # 1 jour

    mask_pos = freqs > 0
    periodes_fft["pres"] = 1.0 / freqs[mask_pos]
    amplitudes_fft["pres"] = np.abs(fft_vals[mask_pos]) * 2.0 / N["pres"]
    mask["pres"] = mask_pos

    # plot spectre (zoom synoptique 2-15 jours)
    per = periodes_fft["pres"]
    amp = amplitudes_fft["pres"]
    zoom = (per >= 2) & (per <= 15)

    plt.figure(figsize=(8,4))
    plt.plot(per[zoom], amp[zoom])
    plt.xlabel("Période (jours)")
    plt.ylabel("Amplitude")
    plt.title("Spectre FFT (synoptique 2–15j) – pres")
    plt.grid()
    plt.savefig("Spectres/pres_spectre_synoptique.png", dpi=150)
    plt.show()
     




# identification des cycles

def trouver_pics(amplitudes, periodes, per_min, per_max, n=5):
    # n le nombre des pics dominants qu'on souhaire recuperer
    mask = (periodes >= per_min) & (periodes <= per_max)
    amps = amplitudes[mask]
    pers = periodes[mask]
    # tri croissant et inversion pour avoir les indices des plus gros amplitudes
    indices = np.argsort(amps)[::-1][:n]
    return [(pers[i], amps[i]) for i in indices if amps[i] > np.mean(amps)]

def cycles(var):
    global amplitudes_fft, periodes_fft, N

    print("\n--- Cycles detectes pour "+var +"---")

    pics_annuel = trouver_pics(amplitudes_fft[var], periodes_fft[var], 300, 400)
    print("\nCycle annuel:")
    for p, a in pics_annuel:
        print("  %.1f jours (amp: %.2f)" % (p, a))

    pics_syno = trouver_pics(amplitudes_fft[var], periodes_fft[var], 3, 60, n=8)
    print("\nCycles synoptiques (meteo):")
    for p, a in pics_syno:
        print("  %.1f jours" % p)
    print("-> entre %.0f et %.0f jours" % (min([p for p,a in pics_syno]), max([p for p,a in pics_syno])))

    pics_pluri = trouver_pics(amplitudes_fft[var], periodes_fft[var], 400, N[var]/2, n=5)
    print("\nCycles pluriannuels:")
    for p, a in pics_pluri:
        print("  %.1f jours = %.1f ans" % (p, p/365))
    print("-> entre %.1f et %.1f ans" % (min([p for p,a in pics_pluri])/365, max([p for p,a in pics_pluri])/365))

    # affichage

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # tout le spectre
    axes[0,0].plot(periodes_fft[var][periodes_fft[var] < 3000], amplitudes_fft[var][periodes_fft[var] < 3000])
    axes[0,0].axvline(x=365, color='r', linestyle='--')
    axes[0,0].set_xlabel("Période (jours)")
    axes[0,0].set_title("Spectre complet "+var)
    axes[0,0].grid()

    # que les cycles synoptiques
    mask_syno = (periodes_fft[var] >= 3) & (periodes_fft[var] <= 60)
    axes[0,1].plot(periodes_fft[var][mask_syno], amplitudes_fft[var][mask_syno], 'g-')
    axes[0,1].set_xlabel("Période (jours)")
    axes[0,1].set_title("Cycles synoptiques pour "+var)
    axes[0,1].grid()

    # que les cycles pluriannuels
    mask_pluri = periodes_fft[var] > 365
    axes[1,0].plot(periodes_fft[var][mask_pluri]/365, amplitudes_fft[var][mask_pluri], 'm-')
    axes[1,0].set_xlabel("Période (années)")
    axes[1,0].set_title("Cycles pluriannuels pour "+var)
    axes[1,0].grid()

    # vision plus proche d'un cycle annuel
    mask_an = (periodes_fft[var] >= 100) & (periodes_fft[var] <= 500)
    axes[1,1].plot(periodes_fft[var][mask_an], amplitudes_fft[var][mask_an])
    axes[1,1].axvline(x=365, color='r', linestyle='--')
    axes[1,1].set_xlabel("Période (jours)")
    axes[1,1].set_title("Zoom cycle annuel")
    axes[1,1].grid()

    plt.tight_layout()
    plt.savefig('Cycles/'+var+'_cycles.png', dpi=150)
    plt.show()
    return pics_syno,pics_annuel, pics_pluri

pics_syno=dict()
pics_annuel=dict()
pics_pluri=dict()

for i in ['tavg', 'tmin', 'tmax', 'pres']:
    p1,p2,p3 = cycles(i)
    pics_syno[i] =p1
    pics_annuel[i] =p2
    pics_pluri[i] =p3



# estimation du bruit

# diff entre donnees/modele
residus = dict()

for i in ['tavg', 'tmin', 'tmax', 'tsun']:
    residus[i] = x_y[i][1] - y_modele[i]


var_signal = dict()
var_bruit = dict()

for i in ['tavg', 'tmin', 'tmax']:
    var_signal[i] = np.var(x_y[i][1])
    var_bruit[i] = np.var(residus[i])

# R^2 = 1 - var bruit/var signal
r2 = dict()
for i in ['tavg', 'tmin', 'tmax']:
    r2[i] = 1 - (var_bruit[i] / var_signal[i])

for i in ['tavg', 'tmin', 'tmax']:
    print("\n--- Bruit pour " + i + " ---")
    print("Variance signal: %.2f" % var_signal[i])
    print("Variance bruit: %.2f" % var_bruit[i])
    print("R2 = %.1f%%" % (r2[i]*100))
    print("Donc %.1f%% de bruit" % ((1-r2[i])*100))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0,0].plot(x_y[i][0], residus[i], 'g-', linewidth=0.3)
    axes[0,0].axhline(y=0, color='r', linestyle='--')
    axes[0,0].set_xlabel("Jours")
    axes[0,0].set_title("Residus pour " + i)
    axes[0,0].grid()

    # les histogrames pour reprensenter la distribution
    axes[0,1].hist(residus[i], bins=50, color='green', alpha=0.7)
    axes[0,1].set_xlabel("Résidu (°C)")
    axes[0,1].set_title("Distribution pour "+var+" (sigma = %.2f)" % np.std(residus[i]))
    axes[0,1].grid()

    # autocorrelation de residus pour verifier si on a bien trouver tous les cycles
    lags = np.arange(0, 365)
    autocorr = np.correlate(residus[i] - np.mean(residus[i]), residus[i] - np.mean(residus[i]), mode='full')
    autocorr = autocorr[len(autocorr)//2:len(autocorr)//2 + len(lags)]
    autocorr = autocorr / autocorr[0]

    axes[1,0].plot(lags, autocorr)
    axes[1,0].axhline(y=0, color='r', linestyle='--')
    axes[1,0].set_xlabel("Décalage (jours)")
    axes[1,0].set_title("Autocorrélation residus pour "+ var)
    axes[1,0].grid()

    fft_bruit = np.fft.fft(residus[i] - np.mean(residus[i]))
    amp_bruit = np.abs(fft_bruit[mask[i]]) * 2 / N[i]
    axes[1,1].plot(periodes_fft[i][periodes_fft[i] < 400], amp_bruit[periodes_fft[i] < 400], 'g-')
    axes[1,1].set_xlabel("Période (jours)")
    axes[1,1].set_title("Spectre du bruit pour " + var)
    axes[1,1].grid()

    plt.tight_layout()
    plt.savefig('Bruits/' + i+ '_bruit.png', dpi=150)
    plt.show()


# tendance climatique


# remplissage des valeurs selon annees

def tendances(var):
    annees = {}
    for row in rows:
        if row[var] is not None and row[var] != '':
            an = row['time'].year
            if an not in annees:
                annees[an] = []
            annees[an].append(float(row[var]))

    moy_an = [(an, np.mean(t)) for an, t in sorted(annees.items())]
    x_an = [m[0] for m in moy_an]
    y_an = [m[1] for m in moy_an]

    # on utilise une regression lineaire pour etudier la tendance climatique

    slope, intercept, r, p, err = linregress(x_an, y_an)

    print("\n--- Tendance climatique ---")
    print("Rechauffement: +%.2f C/decennie" % (slope*10))

    plt.figure(figsize=(10,4))
    plt.scatter(x_an, y_an, alpha=0.7)
    plt.plot(x_an, np.array(x_an)*slope + intercept, 'r-', linewidth=2)
    plt.xlabel("Année")
    plt.ylabel("Temp moyenne (°C)")
    plt.title("Evolution " + var+  " annuelle (+%.2f C/decennie)" % (slope*10))
    plt.grid()
    plt.savefig('Tendances/' + var +'_tendance.png', dpi=150)
    plt.show()


    print("\n--- Resume pour " + var+ " ---")
    print("Cycle annuel: ~365j, amplitude %.1fC" % amplitude[var])
    print("Cycles synoptiques: %d-%d jours" % (min([p for p,a in pics_syno[var]]), max([p for p,a in pics_syno[var]])))
    print("Cycles pluriannuels: %.1f-%.1f ans" % (min([p for p,a in pics_pluri[var]])/365, max([p for p,a in pics_pluri[var]])/365))
    print("Bruit: %.0f%%" % ((1-r2[var])*100))
    print("Rechauffement: +%.2fC/decennie" % (slope*10))

for i in ['tavg', 'tmin', 'tmax']:
    tendances(i)


