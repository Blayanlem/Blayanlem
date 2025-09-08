import os
import numpy as np

# =============================
# === Constants
# =============================
HARTREE_TO_EV = 27.2114


# =============================
# === Utility functions
# =============================
def get_homo_lumo(logfile):
    """Parse Gaussian log to get HOMO/LUMO indices."""
    homo, lumo, occ_count = None, None, 0
    with open(logfile, "r") as f:
        for line in f:
            if "Alpha  occ. eigenvalues" in line:
                values = line.split("--")[-1].split()
                occ_count += len(values)
            elif "Alpha virt. eigenvalues" in line and homo is None:
                homo = occ_count
                lumo = homo + 1
                break
    if homo is None or lumo is None:
        raise ValueError(f"Could not find HOMO/LUMO in {logfile}")
    return homo, lumo


def get_n_basis(logfile):
    """Parse Gaussian log to get NBasis value."""
    with open(logfile, "r") as f:
        for line in f:
            if "NBasis=" in line:
                return int(line.split()[1])
    raise ValueError(f"Could not find NBasis in {logfile}")


def read_lower_triangular_matrix(filename, skip_lines=3):
    """Read lower triangular matrix (Fock or Overlap) from Gaussian file."""
    elements = []
    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            if idx >= skip_lines:
                elements.extend(map(float, line.split()))
    return elements


def form_symmetric_matrix(elements, dim):
    """Reconstruct full symmetric matrix from lower-triangular elements."""
    mat = np.zeros((dim, dim))
    k = 0
    for i in range(dim):
        for j in range(i + 1):
            mat[i, j] = elements[k]
            k += 1
    return mat + np.triu(mat.T, 1)


def read_orbitals(filename, n_basis, coeffs_per_line=5, coeff_length=15):
    """
    Read MO coefficients from Gaussian punch file.
    Returns a dict {orbital_index: [coeffs]}.
    """
    orbitals = {}
    orb_count = -1
    orb_flag = 0
    n_orb_lines = (n_basis // coeffs_per_line) + 1

    with open(filename, 'r') as file:
        next(file)  # skip first line
        for line in file:
            cols = line.split()
            if orb_flag > 0:
                for i in range(coeffs_per_line):
                    coeff_str = line[i * coeff_length:(i + 1) * coeff_length].strip()
                    if coeff_str and 'Alpha' not in line:
                        if 'D' in coeff_str:  # Fortran D-notation
                            base, power = coeff_str.split('D')
                            coeff_str = f"{float(base)}e{int(power)}"
                        orbitals[orb_count].append(float(coeff_str))
                orb_flag += 1
                if orb_flag == n_orb_lines + 1:
                    orb_flag = 0
            if len(cols) > 1 and cols[1] == 'Alpha':
                orb_count = int(cols[0])
                orbitals[orb_count] = []
                orb_flag = 1
    return orbitals


def parse_param_file(param_file):
    """Parse paramfile.txt for orbital ranges and basis functions."""
    with open(param_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    orbitalsA_begin, orbitalsA_end = map(int, lines[0].split('=')[1].split())
    orbitalsB_begin, orbitalsB_end = map(int, lines[1].split('=')[1].split())
    nBasisFunctsA = int(lines[2].split('=')[1])
    nBasisFunctsB = int(lines[3].split('=')[1])
    return nBasisFunctsA, nBasisFunctsB, orbitalsA_begin, orbitalsA_end, orbitalsB_begin, orbitalsB_end


def parse_in_file(in_file):
    """Parse infile.in for overlap, fock, and MO files."""
    with open(in_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
    fragA = lines[2].split('-')[-1]
    fragB = lines[3].split('-')[-1]
    return lines[0], lines[1], lines[2], lines[3], fragA, fragB


def compute_coupling(orbCoeffsA, orbCoeffsB, O, F, nA, nB, orbA, orbB):
    """Compute electronic coupling between fragment A and B orbitals."""
    # Fragment A
    Saa = sum(orbCoeffsA[orbA][i]*O[i,j]*orbCoeffsA[orbA][j] for i in range(nA) for j in range(nA))
    Eaa = sum(orbCoeffsA[orbA][i]*F[i,j]*orbCoeffsA[orbA][j] for i in range(nA) for j in range(nA))
    Eaaev = Eaa * HARTREE_TO_EV

    # Fragment B
    Sbb = sum(orbCoeffsB[orbB][i]*O[i+nA, j+nA]*orbCoeffsB[orbB][j] for i in range(nB) for j in range(nB))
    Ebb = sum(orbCoeffsB[orbB][i]*F[i+nA, j+nA]*orbCoeffsB[orbB][j] for i in range(nB) for j in range(nB))
    Ebbev = Ebb * HARTREE_TO_EV

    # Cross terms
    Sab = sum(orbCoeffsA[orbA][i]*O[j+nA, i]*orbCoeffsB[orbB][j] for i in range(nA) for j in range(nB))
    Eab = sum(orbCoeffsA[orbA][i]*F[j+nA, i]*orbCoeffsB[orbB][j] for i in range(nA) for j in range(nB))
    Eabev = Eab * HARTREE_TO_EV

    # Coupling
    tABev = (Eabev - ((Eaaev+Ebbev)*Sab/2)) / (1 - Sab*Sab)
    return Saa, Eaa, Eaaev, Sbb, Ebb, Ebbev, Sab, Eab, Eabev, tABev


# =============================
# === Coupling driver
# =============================
def coupling(argv):
    """Main coupling workflow: read input, compute couplings, write outputs."""
    in_file = argv[1]
    param_file = argv[2]
    out_prefix = argv[3]

    out_full = out_prefix + '.full'
    out_short = out_prefix + '.short'

    overlap_file, fock_file, molOrbFileA, molOrbFileB, fragA, fragB = parse_in_file(in_file)
    nA, nB, orbitalsA_begin, orbitalsA_end, orbitalsB_begin, orbitalsB_end = parse_param_file(param_file)

    fock_elements = read_lower_triangular_matrix(fock_file, skip_lines=3)
    F = form_symmetric_matrix(fock_elements, nA+nB)

    overlap_elements = read_lower_triangular_matrix(overlap_file, skip_lines=3)
    O = form_symmetric_matrix(overlap_elements, nA+nB)

    orbCoeffsA = read_orbitals(molOrbFileA, nA)
    orbCoeffsB = read_orbitals(molOrbFileB, nB)

    with open(out_full, 'w') as fOut, open(out_short, 'w') as fOut2:
        print(f"=== Coupling Calculation for Fragment A ({fragA}) vs Fragment B ({fragB}) ===", file=fOut)
        print(f"=== Coupling Calculation for Fragment {fragA} vs {fragB} ===", file=fOut2)

        for orbA in range(orbitalsA_begin, orbitalsA_end + 1):
            for orbB in range(orbitalsB_begin, orbitalsB_end + 1):
                Saa, Eaa, Eaaev, Sbb, Ebb, Ebbev, Sab, Eab, Eabev, tABev = compute_coupling(
                    orbCoeffsA, orbCoeffsB, O, F, nA, nB, orbA, orbB
                )
                print(f"t({orbA},{orbB})= {tABev} eV", file=fOut)
                print(f"{fragA}-{fragB}: t({orbA},{orbB})= {tABev} eV", file=fOut2)


# =============================
# === File preparation
# =============================
def make_input_files(system_dir):
    """Create infile.in and paramfile.txt inside a folder."""
    molA, molB = os.path.basename(system_dir).split("_")

    infile = os.path.join(system_dir, "infile.in")
    paramfile = os.path.join(system_dir, "paramfile.txt")

    logA = os.path.join(system_dir, f"{molA}.log")
    logB = os.path.join(system_dir, f"{molB}.log")

    # HOMO/LUMO for each fragment
    homoA, lumoA = get_homo_lumo(logA)
    homoB, lumoB = get_homo_lumo(logB)

    # Basis functions for each fragment
    nA = get_n_basis(logA)
    nB = get_n_basis(logB)

    # Detect .53 and .54 files inside the folder
    file53 = next((f for f in os.listdir(system_dir) if f.endswith(".53")), None)
    file54 = next((f for f in os.listdir(system_dir) if f.endswith(".54")), None)

    if not file53 or not file54:
        raise FileNotFoundError(f"Could not find .53 or .54 files in {system_dir}")

    with open(infile, "w") as f:
        f.write(os.path.join(system_dir, file53) + "\n")
        f.write(os.path.join(system_dir, file54) + "\n")
        f.write(os.path.join(system_dir, f"fort.7-{molA}") + "\n")
        f.write(os.path.join(system_dir, f"fort.7-{molB}") + "\n")

    with open(paramfile, "w") as f:
        f.write(f"orbitalsA= {lumoA} {lumoA+1}\n")
        f.write(f"orbitalsB= {lumoB} {lumoB+1}\n")
        f.write(f"nBasisFunctsA= {nA}\n")
        f.write(f"nBasisFunctsB= {nB}\n")

    return infile, paramfile, molA, molB, nA, nB


def run_one_pair(system_dir):
    """Run coupling for one folder (e.g. 3_1380)."""
    infile, paramfile, molA, molB, nA, nB = make_input_files(system_dir)
    out_prefix = os.path.join(system_dir, f"Output_{molA}_vs_{molB}")
    coupling(['a', infile, paramfile, out_prefix])
    return f"[DONE] {molA}_{molB} (nA={nA}, nB={nB}) → results in {system_dir}"


from concurrent.futures import ProcessPoolExecutor, as_completed

def run_all_pairs(base_dir, max_workers=8):
    """Run coupling in parallel for all folders named like 123_456."""
    folders = [f for f in os.listdir(base_dir) if "_" in f and os.path.isdir(os.path.join(base_dir, f))]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_folder = {executor.submit(run_one_pair, os.path.join(base_dir, folder)): folder for folder in folders}

        for future in as_completed(future_to_folder):
            folder = future_to_folder[future]
            try:
                msg = future.result()  # this comes from run_one_pair
                print(msg, flush=True)
            except Exception as e:
                print(f"[ERROR] Skipped {folder}: {e}", flush=True)


# =============================
# === Main execution
# =============================
if __name__ == "__main__":
    base_dir = r"D:\HK-4\low\output_gjf_files"
    run_all_pairs(base_dir, max_workers=8)   # adjust workers depending on CPU cores
