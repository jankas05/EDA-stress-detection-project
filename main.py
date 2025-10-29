import wfdb
import numpy
import numpy._core.numeric as _nx


EDA = [4]

def custom_split(ary, number_of_segments:int, number_of_entries:int):
        """
        Splits an array into multiple arrays. 

        Please refer to the ''split'' documentation in the numpy library. The only difference
        between these functions is that this function splits an array into number_of_segments
        sub arrays with all except the last one having number_of_entries elements
        """
        Ntotal = len(ary)
        Nsections = number_of_segments
        if Nsections <=0:
                raise ValueError('number selection must be larger then 0.') from None
        section_sizes = number_of_entries
        
        #calculate points, where the array schould be split.
        div_points = []
        for i in range(Ntotal):
            div_points.append(i * section_sizes)

        #do the splitting
        sub_arys = []
        sary = _nx.swapaxes(ary,0,0)
        for i in range(Nsections):
                st = div_points[i]
                try:
                    end = div_points[i+1]
                    sub_arys.append(_nx.swapaxes(sary[st:end], 0, 0))
                except IndexError: #last division point could be out of bound
                    end = Ntotal -1 #do basically nothing

        return sub_arys

def extract_annotation(record:str, extension:str):
      """
      Extract annotation data from an WFDB annotation file
      """
      annotation = wfdb.rdann(record, extension)
      return annotation.sample

def remove_signal_segments(segmented_signal,indices):
      """
      Remove the parts of the signal, where physical stress was expected
      """
      new_signal =[]
      for i in indices:
            new_signal += segmented_signal[i[0]:i[1]]
      return new_signal

def segment_signal(record:str,channel:int, segment_length_s:int):
        """
        Segment a WFDB record of a given channel into given lengths in seconds
        """
        
        #read the signal and extract basic data and annotations
        signal,signal_data = wfdb.rdsamp(record_name=record, channels=channel,warn_empty=True)
        extract_annotation(record=record,extension="atr")
        sampling_frequency = signal_data["fs"]
        signal_length = signal_data["sig_len"]

        #calculate the number of segments and the length of each segment
        entries_in_segment = int(sampling_frequency * segment_length_s)
        number_of_segments = int(signal_length / entries_in_segment)
        #if ((signal_length%entries_in_segment) !=0): number_of_segments+=1

        #split the data using the custom splitting algorithm
        segmented_signal = custom_split(signal,number_of_segments,entries_in_segment)

        return segmented_signal
        



subject_1 = segment_signal(record="data/Subject1/Subject1_AccTempEDA",channel=EDA, segment_length_s=30)
for i in range(75):
      assert(subject_1[i].size == 240)
subject_1_non_physical = remove_signal_segments(subject_1, [[0,10],[20,len(subject_1)]])
print(len(subject_1_non_physical))
assert(subject_1[0][0] == subject_1_non_physical[0][0])

record = wfdb.rdrecord("data/Subject1/Subject1_AccTempEDA")
annotation = wfdb.rdann("data/Subject1/Subject1_AccTempEDA",'atr')
print(annotation.sample)
#print(len(annotation.sample))
#wfdb.plot_wfdb(record=record,annotation=annotation,plot_sym=True,time_units="minutes",title="test-Subject1")
