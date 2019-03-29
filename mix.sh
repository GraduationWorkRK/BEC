#!/bin/sh

##### parameters #####
workdir=$(pwd)

outfile="data_mixed.txt"

##### usage #####
function usage_exit() {
cat << EOF
Usage: $(echo $(basename $0)) [option] param
  param:
  option:

EOF
exit 1
}
set -e

OPT=$(getopt -o h \
              -l help \
              -- "$@")
[ $? != 0 ] && usage_exit

eval set -- "$OPT"
while true; do
  case $1 in
    -h|--help) usage_exit;;
    --) shift; break;;
  esac
  shift
done
shift $(( OPTIND -1 ))

##### error handling #####
if [[ -f $outfile ]]; then
  echo "overwrite $outfile? [yes/no]"
  read ans
  if [[ $ans == "yes" ]]; then
    rm $outfile
  else
    echo "script stopped"
    usage_exit
  fi
fi

##### main method #####

echo "start: $(date)"
for c1 in $(seq -1.0 0.02 1.0); do
  sc1=$(awk 'BEGIN{printf "%+6.3f", '$c1'}')
  for c2 in $(seq -1.0 0.02 1.0); do
    sc2=$(awk 'BEGIN{printf "%+6.3f", '$c2'}')
    file=data_${sc1}_${sc2}.txt
    if [[ -f $file ]]; then
      cat $file >> $outfile
    else
      echo "NO FILE: $file"
    fi
  done
done
echo "finish: $(date)"
