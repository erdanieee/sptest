echo ${1}
version=${1}

if [ ${version} -eq 2 ]; then
    folder="samples"
else
    folder="projections"
fi

echo ${folder}

rsync -r -a -v -e ssh --include="*/" --include="*.Q" --exclude="*"  --delete cloucera@gattaca1:/mnt/lustre/scratch/CBRA/projects/CSVS/spanishTest/v${version}.0/machine_learning/${folder}/ /data/projects/spanishTest/v${version}  
scp cloucera@gattaca1:/mnt/lustre/scratch/CBRA/projects/CSVS/spanishTest/v${version}.0/plink.26.Q /data/projects/spanishTest/v${version}/ 
#`/mnt/lustre/scratch/CBRA/projects/CSVS/spanishTest/v2.0/samples`