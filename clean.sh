#!/bin/bash

BASE='./experiments'

for project_dir in `ls $BASE`
do
	list=('models' 'training_states')
	for dir in ${list[*]}
	do
		declare -a all_file
		count=0
		for i in `ls -tr $BASE'/'$project_dir'/'$dir`
		do
		    all_file[${count}]=${i}
		    count=$(expr ${count} + 1)
		done
		if [ $count -gt 51 ]
		then
			count=`expr $count - 51`
			i=0
			while [ $i -lt $count ]
			do
				rm $BASE'/'$project_dir'/'$dir'/'${all_file[$i]}
				let i++
			done

		fi
	done
done

