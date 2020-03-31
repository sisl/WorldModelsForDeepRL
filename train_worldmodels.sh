startIter=0
numIters=1

for i in `seq 0 $numIters` 
do
	iter=$((startIter+i))
	echo "Start iteration $iter"
	for j in `seq 1 10` 
	do
		echo "Generating 100 samples - batch $j"
		python3 generate_data.py --nsamples 100 --iternum $iter
	done

	echo "Training VAE --iternum $iter"
	python3 train_vae.py --epochs 150 --iternum $iter
	echo "Training MDRNN --iternum $iter"
	python3 train_mdrnn.py --epochs 50 --iternum $iter
	echo "Training Controller --iternum $iter"
	bash train_controller.sh 4 $iter
done
