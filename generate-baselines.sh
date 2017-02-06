#echo -e  "\nGreedy Rouge L Train\n"
#python generate-greedy-rouge-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --size 250 --order L \
#    --rouge RELEASE-1.5.5 \
#    --split train --output summaries/train/greedy250RL
#
#echo -e  "\nGreedy Rouge 1 Train\n"
#python generate-greedy-rouge-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --size 250 --order 1 \
#    --rouge RELEASE-1.5.5 \
#    --split train --output summaries/train/greedy250R1
#
#echo -e  "\nGreedy Rouge 2 Train\n"
#python generate-greedy-rouge-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --size 250 --order 2 \
#    --rouge RELEASE-1.5.5 \
#    --split train --output summaries/train/greedy250R2
#
#echo -e  "\nReference Train\n"
#python generate-reference-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --split train --output summaries/train/reference
#
#echo -e  "\nLead Train\n"
#python generate-lead-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --split train --output summaries/train/lead3 --lead 3
#
echo -e  "\nBackbone Train\n"
python generate-backbone-summaries.py \
    --corpus dailymail --data-path neuralsum/neuralsum \
    --alignments-path data/alignments \
    --split train --output summaries/train/backbone

#echo -e  "\nGreedy Rouge L Dev\n"
#python generate-greedy-rouge-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --size 250 --order L \
#    --rouge RELEASE-1.5.5 \
#    --split dev --output summaries/dev/greedy250RL
#
#echo -e  "\nGreedy Rouge L Test\n"
#python generate-greedy-rouge-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --size 250 --order L \
#    --rouge RELEASE-1.5.5 \
#    --split test --output summaries/test/greedy250RL
#
#echo -e  "\nGreedy Rouge 1 Dev\n"
#python generate-greedy-rouge-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --size 250 --order 1 \
#    --rouge RELEASE-1.5.5 \
#    --split dev --output summaries/dev/greedy250R1
#
#echo -e  "\nGreedy Rouge 1 Test\n"
#python generate-greedy-rouge-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --size 250 --order 1 \
#    --rouge RELEASE-1.5.5 \
#    --split test --output summaries/test/greedy250R1
#
#echo -e  "\nGreedy Rouge 2 Dev\n"
#python generate-greedy-rouge-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --size 250 --order 2 \
#    --rouge RELEASE-1.5.5 \
#    --split dev --output summaries/dev/greedy250R2
#
#echo -e  "\nGreedy Rouge 2 Test\n"
#python generate-greedy-rouge-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --size 250 --order 2 \
#    --rouge RELEASE-1.5.5 \
#    --split test --output summaries/test/greedy250R2
#
#echo -e  "\nReference Dev\n"
#python generate-reference-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --split dev --output summaries/dev/reference
#
#echo -e  "\nLead Dev\n"
#python generate-lead-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --split dev --output summaries/dev/lead3 --lead 3
#
#echo -e  "\nBackbone Dev\n"
#python generate-backbone-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --alignments-path data/alignments \
#    --split dev --output summaries/dev/backbone
#
#echo -e  "\nReference Test\n"
#python generate-reference-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --split test --output summaries/test/reference
#
#echo -e  "\nLead Test\n"
#python generate-lead-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --split test --output summaries/test/lead3 --lead 3
#
#echo -e  "\nBackbone Test\n"
#python generate-backbone-summaries.py \
#    --corpus dailymail --data-path neuralsum/neuralsum \
#    --alignments-path data/alignments \
#    --split test --output summaries/test/backbone
#
#
#
