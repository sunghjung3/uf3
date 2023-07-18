from ase.io import read, write


def outcar2traj(outcar, traj):
    atoms = read(outcar, index=":", format="vasp-out")
    write(traj, atoms)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_traj.py [OUTCAR] [output.traj]")
        sys.exit(1)
    outcar = sys.argv[1]
    traj = sys.argv[2]
    outcar2traj(outcar, traj)
