## This script sections a file into individual vowels, vowels having been marked  in some tier using some uniform label such as "v".
##  Each vowel is saved as an individual Praat Sound file with the name of the original file plus a number indicating which vowel is being saved.
## 25 milliseconds is included to either side of the vowel, to make sure that a formant object includes the entire vowel (25 ms. frames).

##  Specify the label you've used to mark your vowels here:
label$ = "ə"

deleteFile: "/Users/nhogan/Desktop/languages/en/info.txt"

##  Specify the tier where the vowels are labeled here:
tier_number = 3

##  Specify the directory where your sound files and accompanying textgrids are located:
directory$ = "/Users/nhogan/Desktop/languages/en/grid_and_wav/"

##  Specify the directory where you want the segmented vowel sound files to be stored:
outdir$ = "/Users/nhogan/Desktop/SpectrogramTest/schwa/"

##  Specify what file extension your sound files end in (.wav, .aiff...)
extension$ = ".wav"

clearinfo
Create Strings as file list... list 'directory$'*'extension$'
number_of_files = Get number of strings

for a from 1 to number_of_files
     select Strings list
     current_sound$ = Get string... 'a'
     Read from file... 'directory$''current_sound$'
     current_sound$ = selected$("Sound")
     Read from file... 'directory$''current_sound$'.TextGrid
     number_vowels = Count labels... 3 'label$'
     Extract tier... 'tier_number'
     current_tier = selected ("IntervalTier")
     Get starting points... 'label$'
     starting_points = selected ("PointProcess")
     select 'current_tier'
     Get end points... 'label$'
     end_points = selected ("PointProcess")
     for i from 1 to number_vowels
          select 'starting_points'
          start'i' = Get time from index... 'i'
           select 'end_points'
           end'i' = Get time from index... 'i'
     endfor
     select Sound 'current_sound$'
      Edit
      for j from 1 to number_vowels
      editor Sound 'current_sound$'
               start = start'j'
               end = end'j'
				print 'start'
				print  
				print 'end'
				print , 
				appendFileLine: "/Users/nhogan/Desktop/languages/en/info.txt", current_sound$, " ", 'start', " ", 'end'
               Select... 'start' 'end'
               Extract selection
        endeditor
		## run the first, then the second
        ## Write to binary file... 'outdir$''current_sound$'-'j'.Sound
		## Write to WAV file... 'outdir$''current_sound$'-'j'.Sound
     endfor
     select all
     minus Strings list
     Remove
endfor
select all
Remove
print All files have been segmented!  Have a nice day!