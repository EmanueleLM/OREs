import argparse
import copy as cp
import numpy as np
import sys
import time
import tensorflow as tf
import enchant
from keras import backend as K
from tensorflow.keras.models import load_model

import pickle

from IMDB_REL_PATHS import MARABOUPY_REL_PATH, ABDUCTION_REL_PATH, KERAS_REL_PATH, FROZEN_REL_PATH, DATA_SAMPLES, EMBEDDINGS_REL_PATH, TRAIN_REL_PATH, RESULTS_PATH
sys.path.append(MARABOUPY_REL_PATH)
# import Marabou
from maraboupy import Marabou
from maraboupy import MarabouUtils, MarabouCore
# import abduction algorithms
sys.path.append(ABDUCTION_REL_PATH)
from abduction_algorithms_HS_cost import logger, freeze_session, Entails, smallest_explanation_with_cost
# import embeddings relative path
sys.path.append(EMBEDDINGS_REL_PATH)
sys.path.append(TRAIN_REL_PATH)
from glove_utils import load_embedding, pad_sequences

# routines for adversarial attacks (fgsm, pgd, ..)
import adversarial_attacks as adv
prefix = ""
emb_dims = 5
input_len = emb_dims*25
num_words = int(input_len/emb_dims)
model_input_shape = (1, emb_dims*num_words)

tf_model_path = KERAS_REL_PATH + 'fc-25inp-64hu-keras-IMDB-5d.h5'.format(prefix, num_words, emb_dims)

# Load model and test the input_ review
model = load_model(tf_model_path, compile=False)

# Load embedding
EMBEDDING_FILENAME = EMBEDDINGS_REL_PATH+'custom-embedding-IMDB.{}d.txt'.format(emb_dims)
word2index, _, index2embedding = load_embedding(EMBEDDING_FILENAME)
embedding = lambda W : np.array([index2embedding[word2index[w]] for w in W]).reshape(model_input_shape)

predictions = []
inputs = ['# please give this one a miss br br kristy swanson and the rest of the cast rendered terrible performances the show is flat flat flat br br i dont know how michael madison could have allowed this one on his plate he almost seemed to know this wasnt going to work out and his performance was quite lacklustre so all you madison fans give this a miss',
  '# i had always wanted to see this film and the first three fourths proved i hadnt waited in vain but what the hell happened in the end i mean dont get me wrong i liked the film it definitely made me nostalgic of the realistic unique nyc of the 80s that we have lost thanks to giuliani but its missing another half hour',
  '# i saw this movie many years ago and just for kicks decided to rent it and watch it again the plot is a carbon copy from fright night i did like the hairy vampire and the bug eating driver otherwise it was not good at all',
  '# ive seen foxy brown coffy friday foster bucktown and black mama white mama of these this is pam griers worst movie poor acting bad script boring action scenes theres just nothing there avoid this and rent friday foster coffy or foxy brown instead',
  '# im not a fan of scratching but i really dug this movie it gave me a real insight into a world i never had a clue existed and what else is a documentary for funny clever hip just like previous film hype about the grunge music scene',
  '# tim curry was the reason i wanted to see this film and while of course his appearance is always entertaining hes basically wasted in it the rest of the cast doesnt fare too well either in this remake of an early zombie movie that has extremely graphic effects that are totally unnecessary to quote a popular axiom sometimes less is more',
  '# the most striking feature about this well acted film is the almost surreal br br images of the era and time it was shot it i could sense the time and moments were stark and very real even the language was so well chosen its all too often when colloquialisms of todays world are carelessly used in movies about br br another time and place',
  '# the premise of this film is the only thing worthwhile it is very poorly made but the idea was clever if not entirely original its a shame the other aspects of the film werent better the acting is especially bad',
  '# for the most part romance films were never my cup of tea but valley girl is one of the few romance films i not only could sit through but actually enjoy nicholas cage is great in his first role and deborah foreman is cute beyond belief there are some side stories that tend to become muddled but not enough to diminish this film',
  '# this movie is one of the worst remakes i have ever seen in my life the acting is laughable and corman has not improved his piranhas any since 1978 90 of the special effects are lifted from piranha 1978 up from the depths 1979 and humanoids from the deep 1979 it makes piranha ii the spawning look like it belongs on the american film institute list',
  '# after all these years of solving crimes you wouldve expected criminals to know that they cant afford making mistakes with him especially not with regards to talking much this time br br columbo goes to college and actually explains his entire technique but for some reason the murderer still doesnt pay enough attention however this still creates wonderful scenes and delightful dialogues',
  '# i gave this a 2 and it only avoided a 1 because of the occasional unintentional laugh the film is excruciatingly boring and incredibly cheap its even worse if you know anything at all about the fantastic four',
  '# a few words for the people here in cine club the worst crap ever seen on this honorable cinema a very poor script a very bad actors and a very bad movie dont waste your time looking this movie see the very good or any movie have been good commented by me say no more',
  '# i think its about 3 years ago when i saw this movie accidentally i revisited the info site for it here and immediately i felt good again i remembered seeing this movie and loving life again it showed me i could find love and what do you know i have a boyfriend for a year and a half now and love is definitely there',
  '# big fat liar is what you get when you combine terrific writing great production and an emphasis on clever ideas over adolescent pap the two stars work great together and what can i say amanda bynes shines putting and lee majors in the film were brilliant touches watch this film with your kids if you dont laugh throughout it you must not have been paying attention',
  '# another great performance by kiefer sutherland i love his movies because he always plays his role very well for a low budget film this was done very good and kept me on the edge the whole time i love these type of movies and i was glad i caught it on ill be buying the dvd or tape for sure br br 9 10',
  '# excellent performances and a solid but not overplayed story helped this movie exceed my expectations this movie was far better than i was expecting after some of the reviews i had read but frankly those reviewers just got it wrong very inspiring and uplifting highly recommended',
  '# took a very good book and completely butchered it nothing was as it should have been some of the best parts of the book were missing including the major point of the whole book simply the worst adaptation of a stephen king novel ever this movie made the mini series for it look good',
  '# not many people have seen this film and it is a shame because it is a work of art the characters are brilliant the dialogue is sensational and the use of themes leaves the audience wondering i truly loved this film and cant wait to see more of matthew georges work',
  '# interesting twist on the vampire yarn fast loud and moody despite my initial fears kris k carries his part reasonably well and snipes aka blade provides a formidable physical presence lots of blood steel silver burning and exploding bodies provide an enjoyable 110 minute distraction if you like the black look of the matrix then blade will appeal to you blade even has a the sequence',
 '# although its been hailed as a comedy drama i found to be mostly depressing its hard to imagine how spike lee could look back affectionately on so much chaos petty cruelty irresponsibility and mean spiritedness',
  '# i chose to watch this film because i am a stephen nichols fan unfortunately i am unhappy with mr nichols choice to do this movie the film was slow badly acted and included some very graphic sex scenes of mr nichols character with a very young woman watch at your own peril',
  '# drivel utter junk the writers must not have read the book or seen david lynchs film not worth wasting your time br br longer does not make better while more in depth then lynchs film it has gross in accuracies and down plays key parts of the story br br a night at the roxbury is more worth your time',
  '# what a waste of talent a very poor semi coherent script cripples this film rather unimaginative direction too some very faint echoes of fargo here but it just doesnt come off',
  '# this movie is really stupid and very boring most of the time there are almost no ghoulies in it at all there is nothing good about this movie on any level just more bad actors pathetically attempting to make a movie so they can get enough money to eat avoid at all costs',
  '# 1st watched 8 31 1996 dir tim robbins very thought provoking and very well done movie on the subject of the death penalty deserved more recognition and publicity than it received',
  '# this is a great movie for all corey feldman fans this movie has a great cast of young actors a group of teens decide to rob a bank to get some quick cash but all goes wrong when a security gets shot and they take hostages',
  '# this movie is an amazing comedy the script is too funny if u watch it more than once you will enjoy it more though the comedy at times is silly but it really makes u laugh salman khan and aamir khan have given justice to their roles after 1994 i have not come across any hindi movie which was as funny as this',
  '# man did this film stink its obvious this film helped spurn hollywoods need to churn out tired sequels to appeal to the masses first of all it came out too quickly and second of all it just didnt have the same hipness which made the original film so successful no new ground was broken and it turned into a rather mundane effort',
  '# similar story line done many times before and this was no improvement br br 15 minutes into this and you should pretty much be able to turn it off the ending was deja vu all over again br br the only morals i could see out of this are stupidity criminals do not equal success if he screwed you before hes gonna do it again',
  '# not an easy film to like at first with both the lead characters quite unlikeable but luckily the heart and soul of the film is paula touching performance which drives the film into uncharted waters and transcends the rather awkward storyline this gives the film a feeling of real truth and makes you think youve seen something special 7 10',
  '# thriller is the greatest music video of all time performed by the greatest artist of all time thriller really sent music videos going and other artists have been trying to copy thriller in one way or another ever since its a thriller',
  '# i first saw this film on hbo around 1983 and i loved it i scoured all of the auction web sites to buy the vhs copy this is a very good suspense movie with a few twists that make it more interesting i dont want to say too much else because if you ever get a chance to see it youll be glad i didnt say too much',
  '# my first thoughts on this film were of using science fiction as a bad way to show naked women not a brilliant story line it had quite a good ending',
  '# this movies shook my will to live why this abomination isnt the bottom 100 list i dont know br br my life was saved by the healing power of danny trejo br br worst movie ever i dare you watch its like a 90 minute collect calling commercial only much much worse i rather watch the blue screen its that bad really',
  '# all that talent but when ya have poor direction and a weak screenplay it doesnt matter who is in a movie very tired attempt at telling a tale which was actually interesting in the beginning but then quickly fell apart toward the end to bad',
  '# this is a really great film in the pulp fiction genre with a touch of film noir thrown in truly one of emma thompsons best performances to date this film has everything its well written well directed beautifully films and has some great performances i dont know why it didnt catch on its spectacular',
  '# well i am the target market i loved it furthermore my husband also a boomer with strong memories of the 60s liked it a lot too i havent read the book so i went into it neutral i was very pleasantly surprised its now on our highly recommended video list br br',
  '# what can i say an excellent end to an excellent series it never quite got the exposure it deserved in asia but by far the best cop show with the best writing and the best cast on televison ever the end of a great era sorry to see you go',
  '# nothing but the void a pleasant one for those who have known the eighties but well quite boring for those who are not interested in it no screenplay in this film but a hero wandering in an underground new york full of and night clubbers it is aimless pointless and naive but not entirely unpleasant',
'# please give this one a miss br br kristy swanson and the rest of the cast rendered terrible performances the show is flat flat flat br br i dont know how michael madison could have allowed this one on his plate he almost seemed to know this wasnt going to work out and his performance was quite lacklustre so all you madison fans give this a miss',
'# i had always wanted to see this film and the first three fourths proved i hadnt waited in vain but what the hell happened in the end i mean dont get me wrong i liked the film it definitely made me nostalgic of the realistic unique nyc of the 80s that we have lost thanks to giuliani but its missing another half hour',
'# i saw this movie many years ago and just for kicks decided to rent it and watch it again the plot is a carbon copy from fright night i did like the hairy vampire and the bug eating driver otherwise it was not good at all',
'# ive seen foxy brown coffy friday foster bucktown and black mama white mama of these this is pam griers worst movie poor acting bad script boring action scenes theres just nothing there avoid this and rent friday foster coffy or foxy brown instead',
'# im not a fan of scratching but i really dug this movie it gave me a real insight into a world i never had a clue existed and what else is a documentary for funny clever hip just like previous film hype about the grunge music scene',
'# tim curry was the reason i wanted to see this film and while of course his appearance is always entertaining hes basically wasted in it the rest of the cast doesnt fare too well either in this remake of an early zombie movie that has extremely graphic effects that are totally unnecessary to quote a popular axiom sometimes less is more',
'# the most striking feature about this well acted film is the almost surreal br br images of the era and time it was shot it i could sense the time and moments were stark and very real even the language was so well chosen its all too often when colloquialisms of todays world are carelessly used in movies about br br another time and place',
'# the premise of this film is the only thing worthwhile it is very poorly made but the idea was clever if not entirely original its a shame the other aspects of the film werent better the acting is especially bad',
'# for the most part romance films were never my cup of tea but valley girl is one of the few romance films i not only could sit through but actually enjoy nicholas cage is great in his first role and deborah foreman is cute beyond belief there are some side stories that tend to become muddled but not enough to diminish this film',
'# this movie is one of the worst remakes i have ever seen in my life the acting is laughable and corman has not improved his piranhas any since 1978 90 of the special effects are lifted from piranha 1978 up from the depths 1979 and humanoids from the deep 1979 it makes piranha ii the spawning look like it belongs on the american film institute list',
'# after all these years of solving crimes you wouldve expected criminals to know that they cant afford making mistakes with him especially not with regards to talking much this time br br columbo goes to college and actually explains his entire technique but for some reason the murderer still doesnt pay enough attention however this still creates wonderful scenes and delightful dialogues',
'# i gave this a 2 and it only avoided a 1 because of the occasional unintentional laugh the film is excruciatingly boring and incredibly cheap its even worse if you know anything at all about the fantastic four',
'# a few words for the people here in cine club the worst crap ever seen on this honorable cinema a very poor script a very bad actors and a very bad movie dont waste your time looking this movie see the very good or any movie have been good commented by me say no more',
'# i think its about 3 years ago when i saw this movie accidentally i revisited the info site for it here and immediately i felt good again i remembered seeing this movie and loving life again it showed me i could find love and what do you know i have a boyfriend for a year and a half now and love is definitely there',
'# big fat liar is what you get when you combine terrific writing great production and an emphasis on clever ideas over adolescent pap the two stars work great together and what can i say amanda bynes shines putting and lee majors in the film were brilliant touches watch this film with your kids if you dont laugh throughout it you must not have been paying attention',
'# another great performance by kiefer sutherland i love his movies because he always plays his role very well for a low budget film this was done very good and kept me on the edge the whole time i love these type of movies and i was glad i caught it on ill be buying the dvd or tape for sure br br 9 10',
'# excellent performances and a solid but not overplayed story helped this movie exceed my expectations this movie was far better than i was expecting after some of the reviews i had read but frankly those reviewers just got it wrong very inspiring and uplifting highly recommended',
'# took a very good book and completely butchered it nothing was as it should have been some of the best parts of the book were missing including the major point of the whole book simply the worst adaptation of a stephen king novel ever this movie made the mini series for it look good',
'# not many people have seen this film and it is a shame because it is a work of art the characters are brilliant the dialogue is sensational and the use of themes leaves the audience wondering i truly loved this film and cant wait to see more of matthew georges work',
'# interesting twist on the vampire yarn fast loud and moody despite my initial fears kris k carries his part reasonably well and snipes aka blade provides a formidable physical presence lots of blood steel silver burning and exploding bodies provide an enjoyable 110 minute distraction if you like the black look of the matrix then blade will appeal to you blade even has a the sequence',
'# although its been hailed as a comedy drama i found to be mostly depressing its hard to imagine how spike lee could look back affectionately on so much chaos petty cruelty irresponsibility and mean spiritedness',
'# i chose to watch this film because i am a stephen nichols fan unfortunately i am unhappy with mr nichols choice to do this movie the film was slow badly acted and included some very graphic sex scenes of mr nichols character with a very young woman watch at your own peril',
'# drivel utter junk the writers must not have read the book or seen david lynchs film not worth wasting your time br br longer does not make better while more in depth then lynchs film it has gross in accuracies and down plays key parts of the story br br a night at the roxbury is more worth your time',
'# what a waste of talent a very poor semi coherent script cripples this film rather unimaginative direction too some very faint echoes of fargo here but it just doesnt come off',
'# this movie is really stupid and very boring most of the time there are almost no ghoulies in it at all there is nothing good about this movie on any level just more bad actors pathetically attempting to make a movie so they can get enough money to eat avoid at all costs',
'# 1st watched 8 31 1996 dir tim robbins very thought provoking and very well done movie on the subject of the death penalty deserved more recognition and publicity than it received',
'# this is a great movie for all corey feldman fans this movie has a great cast of young actors a group of teens decide to rob a bank to get some quick cash but all goes wrong when a security gets shot and they take hostages',
'# this movie is an amazing comedy the script is too funny if u watch it more than once you will enjoy it more though the comedy at times is silly but it really makes u laugh salman khan and aamir khan have given justice to their roles after 1994 i have not come across any hindi movie which was as funny as this',
'# man did this film stink its obvious this film helped spurn hollywoods need to churn out tired sequels to appeal to the masses first of all it came out too quickly and second of all it just didnt have the same hipness which made the original film so successful no new ground was broken and it turned into a rather mundane effort',
'# similar story line done many times before and this was no improvement br br 15 minutes into this and you should pretty much be able to turn it off the ending was deja vu all over again br br the only morals i could see out of this are stupidity criminals do not equal success if he screwed you before hes gonna do it again',
'# not an easy film to like at first with both the lead characters quite unlikeable but luckily the heart and soul of the film is paula touching performance which drives the film into uncharted waters and transcends the rather awkward storyline this gives the film a feeling of real truth and makes you think youve seen something special 7 10',
'# thriller is the greatest music video of all time performed by the greatest artist of all time thriller really sent music videos going and other artists have been trying to copy thriller in one way or another ever since its a thriller',
'# i first saw this film on hbo around 1983 and i loved it i scoured all of the auction web sites to buy the vhs copy this is a very good suspense movie with a few twists that make it more interesting i dont want to say too much else because if you ever get a chance to see it youll be glad i didnt say too much',
'# my first thoughts on this film were of using science fiction as a bad way to show naked women not a brilliant story line it had quite a good ending',
'# this movies shook my will to live why this abomination isnt the bottom 100 list i dont know br br my life was saved by the healing power of danny trejo br br worst movie ever i dare you watch its like a 90 minute collect calling commercial only much much worse i rather watch the blue screen its that bad really',
'# all that talent but when ya have poor direction and a weak screenplay it doesnt matter who is in a movie very tired attempt at telling a tale which was actually interesting in the beginning but then quickly fell apart toward the end to bad',
'# this is a really great film in the pulp fiction genre with a touch of film noir thrown in truly one of emma thompsons best performances to date this film has everything its well written well directed beautifully films and has some great performances i dont know why it didnt catch on its spectacular',
'# well i am the target market i loved it furthermore my husband also a boomer with strong memories of the 60s liked it a lot too i havent read the book so i went into it neutral i was very pleasantly surprised its now on our highly recommended video list br br',
'# what can i say an excellent end to an excellent series it never quite got the exposure it deserved in asia but by far the best cop show with the best writing and the best cast on televison ever the end of a great era sorry to see you go',
'# nothing but the void a pleasant one for those who have known the eighties but well quite boring for those who are not interested in it no screenplay in this film but a hero wandering in an underground new york full of and night clubbers it is aimless pointless and naive but not entirely unpleasant',
  '# please give this one a miss br br kristy swanson and the rest of the cast rendered terrible performances the show is flat flat flat br br i dont know how michael madison could have allowed this one on his plate he almost seemed to know this wasnt going to work out and his performance was quite lacklustre so all you madison fans give this a miss',
  '# i had always wanted to see this film and the first three fourths proved i hadnt waited in vain but what the hell happened in the end i mean dont get me wrong i liked the film it definitely made me nostalgic of the realistic unique nyc of the 80s that we have lost thanks to giuliani but its missing another half hour',
  '# i saw this movie many years ago and just for kicks decided to rent it and watch it again the plot is a carbon copy from fright night i did like the hairy vampire and the bug eating driver otherwise it was not good at all',
  '# ive seen foxy brown coffy friday foster bucktown and black mama white mama of these this is pam griers worst movie poor acting bad script boring action scenes theres just nothing there avoid this and rent friday foster coffy or foxy brown instead',
  '# im not a fan of scratching but i really dug this movie it gave me a real insight into a world i never had a clue existed and what else is a documentary for funny clever hip just like previous film hype about the grunge music scene',
  '# tim curry was the reason i wanted to see this film and while of course his appearance is always entertaining hes basically wasted in it the rest of the cast doesnt fare too well either in this remake of an early zombie movie that has extremely graphic effects that are totally unnecessary to quote a popular axiom sometimes less is more',
  '# the most striking feature about this well acted film is the almost surreal br br images of the era and time it was shot it i could sense the time and moments were stark and very real even the language was so well chosen its all too often when colloquialisms of todays world are carelessly used in movies about br br another time and place',
  '# the premise of this film is the only thing worthwhile it is very poorly made but the idea was clever if not entirely original its a shame the other aspects of the film werent better the acting is especially bad',
  '# for the most part romance films were never my cup of tea but valley girl is one of the few romance films i not only could sit through but actually enjoy nicholas cage is great in his first role and deborah foreman is cute beyond belief there are some side stories that tend to become muddled but not enough to diminish this film',
  '# this movie is one of the worst remakes i have ever seen in my life the acting is laughable and corman has not improved his piranhas any since 1978 90 of the special effects are lifted from piranha 1978 up from the depths 1979 and humanoids from the deep 1979 it makes piranha ii the spawning look like it belongs on the american film institute list',
  '# after all these years of solving crimes you wouldve expected criminals to know that they cant afford making mistakes with him especially not with regards to talking much this time br br columbo goes to college and actually explains his entire technique but for some reason the murderer still doesnt pay enough attention however this still creates wonderful scenes and delightful dialogues',
  '# i gave this a 2 and it only avoided a 1 because of the occasional unintentional laugh the film is excruciatingly boring and incredibly cheap its even worse if you know anything at all about the fantastic four',
  '# a few words for the people here in cine club the worst crap ever seen on this honorable cinema a very poor script a very bad actors and a very bad movie dont waste your time looking this movie see the very good or any movie have been good commented by me say no more',
  '# i think its about 3 years ago when i saw this movie accidentally i revisited the info site for it here and immediately i felt good again i remembered seeing this movie and loving life again it showed me i could find love and what do you know i have a boyfriend for a year and a half now and love is definitely there',
  '# big fat liar is what you get when you combine terrific writing great production and an emphasis on clever ideas over adolescent pap the two stars work great together and what can i say amanda bynes shines putting and lee majors in the film were brilliant touches watch this film with your kids if you dont laugh throughout it you must not have been paying attention',
  '# another great performance by kiefer sutherland i love his movies because he always plays his role very well for a low budget film this was done very good and kept me on the edge the whole time i love these type of movies and i was glad i caught it on ill be buying the dvd or tape for sure br br 9 10',
  '# excellent performances and a solid but not overplayed story helped this movie exceed my expectations this movie was far better than i was expecting after some of the reviews i had read but frankly those reviewers just got it wrong very inspiring and uplifting highly recommended',
  '# took a very good book and completely butchered it nothing was as it should have been some of the best parts of the book were missing including the major point of the whole book simply the worst adaptation of a stephen king novel ever this movie made the mini series for it look good',
  '# not many people have seen this film and it is a shame because it is a work of art the characters are brilliant the dialogue is sensational and the use of themes leaves the audience wondering i truly loved this film and cant wait to see more of matthew georges work',
  '# interesting twist on the vampire yarn fast loud and moody despite my initial fears kris k carries his part reasonably well and snipes aka blade provides a formidable physical presence lots of blood steel silver burning and exploding bodies provide an enjoyable 110 minute distraction if you like the black look of the matrix then blade will appeal to you blade even has a the sequence',
  '# although its been hailed as a comedy drama i found to be mostly depressing its hard to imagine how spike lee could look back affectionately on so much chaos petty cruelty irresponsibility and mean spiritedness',
  '# i chose to watch this film because i am a stephen nichols fan unfortunately i am unhappy with mr nichols choice to do this movie the film was slow badly acted and included some very graphic sex scenes of mr nichols character with a very young woman watch at your own peril',
  '# drivel utter junk the writers must not have read the book or seen david lynchs film not worth wasting your time br br longer does not make better while more in depth then lynchs film it has gross in accuracies and down plays key parts of the story br br a night at the roxbury is more worth your time',
  '# what a waste of talent a very poor semi coherent script cripples this film rather unimaginative direction too some very faint echoes of fargo here but it just doesnt come off',
  '# this movie is really stupid and very boring most of the time there are almost no ghoulies in it at all there is nothing good about this movie on any level just more bad actors pathetically attempting to make a movie so they can get enough money to eat avoid at all costs',
  '# 1st watched 8 31 1996 dir tim robbins very thought provoking and very well done movie on the subject of the death penalty deserved more recognition and publicity than it received',
  '# this is a great movie for all corey feldman fans this movie has a great cast of young actors a group of teens decide to rob a bank to get some quick cash but all goes wrong when a security gets shot and they take hostages',
  '# this movie is an amazing comedy the script is too funny if u watch it more than once you will enjoy it more though the comedy at times is silly but it really makes u laugh salman khan and aamir khan have given justice to their roles after 1994 i have not come across any hindi movie which was as funny as this',
  '# man did this film stink its obvious this film helped spurn hollywoods need to churn out tired sequels to appeal to the masses first of all it came out too quickly and second of all it just didnt have the same hipness which made the original film so successful no new ground was broken and it turned into a rather mundane effort',
  '# similar story line done many times before and this was no improvement br br 15 minutes into this and you should pretty much be able to turn it off the ending was deja vu all over again br br the only morals i could see out of this are stupidity criminals do not equal success if he screwed you before hes gonna do it again',
  '# not an easy film to like at first with both the lead characters quite unlikeable but luckily the heart and soul of the film is paula touching performance which drives the film into uncharted waters and transcends the rather awkward storyline this gives the film a feeling of real truth and makes you think youve seen something special 7 10',
  '# thriller is the greatest music video of all time performed by the greatest artist of all time thriller really sent music videos going and other artists have been trying to copy thriller in one way or another ever since its a thriller',
  '# i first saw this film on hbo around 1983 and i loved it i scoured all of the auction web sites to buy the vhs copy this is a very good suspense movie with a few twists that make it more interesting i dont want to say too much else because if you ever get a chance to see it youll be glad i didnt say too much',
  '# my first thoughts on this film were of using science fiction as a bad way to show naked women not a brilliant story line it had quite a good ending',
  '# this movies shook my will to live why this abomination isnt the bottom 100 list i dont know br br my life was saved by the healing power of danny trejo br br worst movie ever i dare you watch its like a 90 minute collect calling commercial only much much worse i rather watch the blue screen its that bad really',
  '# all that talent but when ya have poor direction and a weak screenplay it doesnt matter who is in a movie very tired attempt at telling a tale which was actually interesting in the beginning but then quickly fell apart toward the end to bad',
  '# this is a really great film in the pulp fiction genre with a touch of film noir thrown in truly one of emma thompsons best performances to date this film has everything its well written well directed beautifully films and has some great performances i dont know why it didnt catch on its spectacular',
  '# well i am the target market i loved it furthermore my husband also a boomer with strong memories of the 60s liked it a lot too i havent read the book so i went into it neutral i was very pleasantly surprised its now on our highly recommended video list br br',
  '# what can i say an excellent end to an excellent series it never quite got the exposure it deserved in asia but by far the best cop show with the best writing and the best cast on televison ever the end of a great era sorry to see you go',
  '# nothing but the void a pleasant one for those who have known the eighties but well quite boring for those who are not interested in it no screenplay in this film but a hero wandering in an underground new york full of and night clubbers it is aimless pointless and naive but not entirely unpleasant']

padded_inputs = []
# Review + <pad>(s)
for input_without_padding in inputs:
    input_without_padding = input_without_padding.lower().split(' ') 
    input_ = input_without_padding[:num_words] + ['<PAD>']*(num_words - len(input_without_padding))
    orig_input = input_.copy()
    padded_inputs.append(orig_input)
    x = embedding(input_)
    input_shape = x.shape
    prediction = model.predict(x)
    input_ = x.flatten().tolist()
    y_hat = np.argmax(prediction)
    c_hat = np.max(prediction)
    predictions.append(y_hat)

pred_dict = dict()
for idx, padded_input in enumerate(padded_inputs):
    for word in padded_input:
        if not word in pred_dict.keys():
            # [0             ,  0               ]
            # bad preds (0's),  good preds (1's)]
            pred_dict[word] = [0,0]
        pred_dict[word][int(predictions[idx])] += 1


for key in pred_dict.keys():
    s = sum(pred_dict[key])
    new_0 = pred_dict[key][0] / s
    new_1 = pred_dict[key][1] / s
    pred_dict[key] = [new_0,new_1]

print(pred_dict)

with open("prec_dict.pkl", "wb") as f:
    pickle.dump(pred_dict,f)
