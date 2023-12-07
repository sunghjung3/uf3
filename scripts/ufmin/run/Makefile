run : clean
	python ufmin.py

clean : clean_vasp
	rm -f model*.json *.pckl *.traj *.out *feature* *.png *.gif

clean_vasp :
	rm -f ase-sort.dat CHG CHGCAR CONTCAR DOSCAR EIGENVAL IBZKPT INCAR OSZICAR OUTCAR PCDAT POTCAR REPORT vasp.out vaspout.h5 vasprun.xml WAVECAR XDATCAR

info :
	@echo "Makefile, myopts.py, POSCAR, postrun_analysis.py, preprocess.py, settings.yaml, ufmin.py"
