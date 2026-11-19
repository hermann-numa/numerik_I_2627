import time
import numpy as np
import matplotlib.pyplot as plt

def compute_and_plot_06(mul_A, mul_pre, cg, pcg, m, tol):
    """
    Diese Routine ruft beide Lösungsverfahren (CG und PCG) auf für ein System
    mit Komplexität m (Größe der Matrix: 2^m+1)
    Die (Zeilensummen-)Norm der Systemmatrix ist aufgabenbedingt stets 4.0.
    """
    #
    #  time und numpy.linalg initialisieren dauert SEHR lang!
    #  Die folgenden Zeilen sind eingefügt, damit die Laufzeit-Tabelle in der
    #  ersten Zeile (m==3) die Initialisierungszeiten der Bibliotheken nicht
    #  mitzählt; bitte NICHT entfernen!
    #
    _ = time.perf_counter()
    _ = np.ones(1)

    #  Aufgabe initialisieren:
    pn = 2**m
    b = -np.ones(pn + 1)
    half_point = int(pn * 0.5 + 0.5)
    b[half_point] = 1.0
    max_iter = pn

    A = 2.0 * np.eye(pn + 1) - np.diag(np.ones(pn), k=-1) - np.diag(np.ones(pn), k=1)    # für die Referenzlösung erzeugen wir die Matrix
    ref_solution = np.linalg.solve(A, b)

    #CG
    cg_tic = time.perf_counter()
    cg_solution, cg_residuals = cg(b, mul_A, tol, max_iter)
    cg_time = time.perf_counter() - cg_tic
    cg_error_norm = np.max(np.abs(ref_solution - cg_solution))

    #PCG
    pcg_tic = time.perf_counter()
    pcg_solution, pcg_residuals = pcg(b, mul_A, mul_pre, tol, max_iter)
    pcg_time = time.perf_counter() - pcg_tic
    pcg_error_norm = np.max(np.abs(ref_solution - pcg_solution))

    #Plots
    cg_xs  = np.arange(1, cg_residuals.shape[0]+1)
    pcg_xs = np.arange(1, pcg_residuals.shape[0]+1)
    xs = np.linspace(-1.0, 1.0, cg_solution.shape[0])
    plt.figure()

    plt.subplot(221)
    plt.title("CG - Residuen", fontsize=12)
    plt.semilogy(cg_xs, cg_residuals, "b-", linewidth=2)

    plt.subplot(223)
    plt.title("CG - Fehler", fontsize=12)
    plt.plot(xs, ref_solution - cg_solution, "r-", linewidth=2)

    plt.subplot(222)
    plt.title("PCG - Residuen", fontsize=12)
    plt.semilogy(pcg_xs, pcg_residuals, "b-", linewidth=2)

    plt.subplot(224)
    plt.title("PCG - Fehler", fontsize=12)
    plt.plot(xs, ref_solution - pcg_solution, "r-", linewidth=2)

    plt.suptitle(f"CG- vs. PCG-Verfahren  @  m = {m},   n = {2**m + 1}", fontsize=16)
    plt.tight_layout()
    
    return (
        cg_solution,
        cg_residuals,
        cg_error_norm,
        cg_time,
        pcg_solution,
        pcg_residuals,
        pcg_error_norm,
        pcg_time,
    )
