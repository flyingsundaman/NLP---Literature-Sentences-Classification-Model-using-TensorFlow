from LitRevSentences.pipeline.training_pipeline import TrainPipeline
from LitRevSentences.pipeline.prediction_pipeline import PredictionPipeline

#train_pipeline = TrainPipeline()

#train_pipeline.run_pipeline()

prediction_pipeline = PredictionPipeline()

LitRev = "To compare the effectiveness of the Alfredson eccentric heel-drop protocol with a `` do-as-tolerated '' protocol for nonathletic individuals with midportion Achilles tendinopathy . The Alfredson protocol recommends the completion of @ eccentric heel drops a day . However , completing this large number of repetitions is time consuming and potentially uncomfortable . There is a need to investigate varying exercise dosages that minimize the discomfort yet retain the clinical benefits . Twenty-eight individuals from outpatient physiotherapy departments were randomized to either the standard ( n = @ ) or the do-as-tolerated ( n = @ ) @-week intervention protocol . Apart from repetition volume , all other aspects of management were standardized between groups . Tendinopathy clinical severity was assessed with the Victorian Institute of Sport Assessment-Achilles ( VISA-A ) questionnaire . Pain intensity was assessed using a visual analog scale ( VAS ) . Both were assessed at baseline , @ weeks , and @ weeks . Treatment satisfaction was assessed at week @ . Adverse effects were also monitored . There was a statistically significant within-group improvement in VISA-A score for both groups ( standard , P = @ ; do as tolerated , P < @ ) and VAS pain for the do-as-tolerated group ( P = @ ) at week @ , based on the intention-to-treat analysis . There was a statistically significant between-group difference in VISA-A scores at week @ , based on both the intention-to-treat ( P = @ ) and per-protocol analyses ( P = @ ) , partly due to a within-group deterioration at week @ in the standard group . There were no statistically significant between-group differences for VISA-A and VAS pain scores at week @ , the completion of the intervention . There was no significant association between satisfaction and treatment groups at week @ . No adverse effects were reported . Performing a @-week do-as-tolerated program of eccentric heel-drop exercises , compared to the recommended @ repetitions per day , did not lead to lesser improvement for individuals with midportion Achilles tendinopathy , based on VISA-A and VAS scores ."

prediction_pipeline.run_pipeline(LitRev=LitRev)

print('Successfully executed pipeline')