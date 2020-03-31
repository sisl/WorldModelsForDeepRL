
numWorkers=$1
iterNum=$2
baseport=$3

# echo "Iter $iterNum for $numWorkers workers"

open_terminal()
{
	script=$1
	if [[ "$OSTYPE" == "darwin"* ]]; then # Running on mac
		osascript <<END 
		tell app "Terminal" to do script "cd \"`pwd`\"; $script; exit"
END
	elif [[ "$OSTYPE" == "linux-gnu" ]]; then # Running on linux
		xterm -e $script $2
	fi
}

run()
{
	ports=()
	for j in `seq 0 $numWorkers` 
	do
		port=$((4000+$baseport+$j))
		ports+=($port)
	done
	port_string=$( IFS=$' '; echo "${ports[*]}" )

	for j in `seq $numWorkers -1 1` 
	do
		open_terminal "python3 -B train_controller.py --iternum $iterNum --tcp_rank $j --tcp_ports $port_string" &
		sleep 0.4
	done
	sleep 4
	python3 -B train_controller.py --iternum $iterNum --tcp_rank 0 --tcp_ports $port_string
}

run
