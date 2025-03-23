# Vocal Tube Estimation 3   

Estimation four tube model length and area about japanese vowels.   

[github repository](https://github.com/shun60s/Vocal-Tube-Estimation3/)  

## usage   

pks2tube5frame is main program.  
It loads a vowel wav file(16Khz mono), LPC analysis, get peaks candidates and pitch(F0) candidate,
exclude outliers using a cubic expression and interpolate about peaks and pitch(F0), and
estimate four tube model length and area by grid search and scipy's optimize.fmin, downhill simplex algorithm.    
  
option:  
- wav_file  input vowel wav file(16Khz mono) path.  
- result_dir the path to store result figures that show estimated tube length and area with frequency response.   
- frame   use previous frame LA0 (estimated length and area) as initial value of scipy's optimize.fmin. specify start frame number. Or, negative value when OFF(default).   
- BPF_out  compute BPF and show frequency response. specify -B if compute BPF frequency response. 

```
python pks2tube5frame.py -w wav/a_1-16k.wav -r result_figure_a -B
```
Human vocal tract moves smoothly and continuously.  
However, LPC analysis peaks candidates and pitch(F0) candidate is sometimes discontinuously.  
They are translated to continuous values using a cubic expression and interpolate.  
 ![figure1](docs/Figure_a_1-16k.png)   

Following estimation result, tube length and area, are inconsistent. Their movement are not smoothly and continuously.  
 ![figure2](docs/Figure_result_figure_a.png)   
These figures are in the result_figure_a folder.  


And then, manually select a reasonable frame for human vowel vocal tract as start frame,  
use previous frame LA0 (estimated length and area) as initial value of scipy's optimize.fmin,  and repeats its operation.  

```
python pks2tube5frame.py -w wav/a_1-16k.wav -r result_figure_a -f 14 -B
```
 ![figure3](docs/Figure_result_figure_a_14.png)   
These movement is almost smoothly and continuously.   
These figures are in the result_figure_a/14 folder.  


## Estimation vocal frequency response by BPF analysis  

Voice BPF output has harmonic structure and it shows only digitized samples of vocal frequency response.  
It's necessary to estimate vocal overall frequency response to know true peaks (formants) by any method.  
Following is vocal frequency response estimation by curve fitting via F0 harmonic frequencies (fundamental and overtones). 

```
python BPF_analysis2.py -w wav/a_1-16k.wav
```

 ![figure4](docs/Figure_curve_fitting_via_F0_harmonic_a_14.png)   



## License    
MIT  
   

