# RedAnDA TS

TWO DIFFERENT EXPERIMENTS CAN BE CONDUCTED. THE OSSE (ANALYSES IN THE SIMULATED FRAMEWORK, WITH SYNTHETIC IN SITU PROFILES) AND THE REAL DATA APPLICATION (ANALYSES OF THE REAL OBSERVATIONS FROM EN4.2.2). THE OSSE FILES ARE REFERED TO WITH THE "OCCIPUT" TERM WHILE THE REAL DATA ONES DOES NOT.

"OBS&GRIDDEDFIELD_GENERATION_[OCCIPUT].IPYNB" THESE NOTEBOOKS ALLOWS TO READ THE OBSERVATIONS AND STANDARDIZE THEIR DEPTH LEVELS AND AVERAGE THE HIGH-FREQUENCY SIGNALS. THE REAL DATA FILE ALSO GENERATE THE INSTRUMENTAL ERROR FILES. THE LEARNING PERIOD DATASETS ARE PREPARED IN THESE NOTEBOOKS. THE EN4 AND IAP PRODUCTS ARE PREPARED FOR THE INTERCOMPARISON.

"RobsGenerator-[PSAL_OCCIPUT ; TEMP-OCCIPUT ; Real-PSAL ; Real ].py" THERE PYTHON SCRIPTS READ THE OBSERVATIONS AND LEARNING PERIOD FILES TO GENERATE THE ERROR OF REPRESENTATIVITY FOR TEMPERATURE, SALINITY, IN THE OSSE AND REAL DATA FRAMEWORKS. THE ERROR OF REPRESENTATIVITY ARE THE VARIANCE OF THE INTERANNUAL ANOMALIES OF THE OBSERVATIONS, WITHIN REGIONAL BOXES, AT EACH STANDARDIZED DEPTH LEVELS

"redAnDA-OCCIPUT-[PSAL ; TEMP&PSAL ; ].py" THERE SCRIPTS READ THE OBSERVATIONS, LEARNING PERIOD, INTRUMENTAL ERROR AND REPRESENTATIVITY ERROR TO PERFORM [UNIVARS, REDANDA TS, UNIVART] ANALYSES, IN THE OSSE.

"redAnDA-[PSAL ; TEMP&PSAL ; ].py" THERE SCRIPTS READ THE OBSERVATIONS, LEARNING PERIOD, INTRUMENTAL ERROR AND REPRESENTATIVITY ERROR TO PERFORM [UNIVARS, REDANDA TS, UNIVART] ANALYSES, IN THE REAL DATA APPLICATION.

"RedAnDA_functions.py" THIS FILE CONTAINS SOME FUNCTIONS USED IN REDANDA, NOTABLY THE ANALOG PREDICTION MODEL

"OI-[TEMP ; PSAL].py" THE OPTIMAL INTERPOLATION READ THE OBSERVATIONS, LEARNING PERIOD, INTRUMENTAL ERROR AND REPRESENTATIVITY ERROR AND PERFORM [TEMPERATURE ; SALINITY] ANALYSIS, IN THE OSSE.

"Analysis_TEMP&PSAL_[OCCIPUT].ipynb" THESE NOTEBOOKS ANALYSE THE RESULTS OF THE DIFFERENTS ANALYSES (OI, UNIVARS, REDANDA TS... FOR THE OSSE. EN4, IAP, UNIVARS AND REDANDA TS FOR THE REAL DATA APPLICATION) AND CREATE THE FIGURES PRESENTED IN OULHEN ET AL. (2025). THE CORRELATION SCORES ARE ALSO GENERATED.
