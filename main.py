import wfdb
import numpy
import numpy._core.numeric as _nx


EDA = [4]

def custom_split(ary:list, number_of_entries:int):
        """
        Splits an array into multiple arrays. 

        Please refer to the ''split'' documentation in the numpy library. The only difference
        between these functions is that this function splits an array into sub arrays with
        a given number of entries. If the last one array would be smaller than the given
        number, it is excluded from the new array.
        """

        Ntotal = len(ary)
        Nsections = int(Ntotal/number_of_entries)
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

def segment_signal(record:str,channel:list, segment_length_s:int):
        """
        Segment a WFDB record of a given channel into given lengths in seconds
        """
        
        #read the signal and extract basic data and annotations
        signal,signal_data = wfdb.rdsamp(record_name=record, channels=channel,warn_empty=True)
        annotation_points = extract_annotation(record=record,extension="atr")
        sampling_frequency = signal_data["fs"]
        signal_length = signal_data["sig_len"]

        #annotation constants for easier readability
        RELAX_ONE_START = annotation_points[0]
        PHYS_START = annotation_points[1]
        RELAX_TWO_START = annotation_points[2]
        COGN_START = annotation_points[3]
        RELAX_THREE_START = annotation_points[5]
        EMOT_START = annotation_points[6]
        RELAX_FOUR_START = annotation_points[7]

        #calculate the length of each segment
        entries_in_segment = int(sampling_frequency * segment_length_s)

        #initialize the arrays for the segments
        non_stress_signal =[]
        stress_signal =[]

        #split the data using the custom splitting algorithm
        non_stress_signal.extend(custom_split( signal[RELAX_ONE_START : PHYS_START],entries_in_segment))
        non_stress_signal.extend(custom_split( signal[RELAX_TWO_START : COGN_START],entries_in_segment))
        stress_signal.extend(custom_split( signal[COGN_START : RELAX_THREE_START], entries_in_segment))
        non_stress_signal.extend(custom_split( signal[RELAX_THREE_START : EMOT_START], entries_in_segment))
        stress_signal.extend(custom_split( signal[EMOT_START : RELAX_FOUR_START], entries_in_segment))
        non_stress_signal.extend(custom_split( signal[RELAX_FOUR_START : ], entries_in_segment))

        return stress_signal, non_stress_signal
        
def group_all_data(directory:str, channel:list, data_count:int, segment_length:int):
      """
      Groups all subject data into stress and non stress segments. 
      """

      #declare needed arrays
      global stress_segments
      global non_stress_segments
      stress_segments = []
      non_stress_segments = []
      
      #go through the subject data and group it into the stress and non stress categories
      for i in range(data_count):
            record_name = directory + "/Subject" + str(i + 1) +"_AccTempEDA"
            temp_stress , temp_non_stress = segment_signal(record=record_name, 
                                                           channel=channel, 
                                                           segment_length_s=segment_length)
            stress_segments.extend(temp_stress)
            non_stress_segments.extend(temp_non_stress)



group_all_data(directory="data", channel=EDA, data_count=20, segment_length=30)


a = len(stress_segments)
b = len(non_stress_segments)
print(a)
print(b)
print(a+b)

for i in range(a):
      assert(len(stress_segments[i]) == 240)
for i in range(b):
      assert(len(non_stress_segments[i]) == 240)
print("finished")
