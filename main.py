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

        returns:
        sub_arys - a numpy array with subarrays of the given length
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
        Extract annotation data from an WFDB annotation file.

        returns:
        annotation.sample - the annotation marks as written in the annotation file

        """

        annotation = wfdb.rdann(record, extension)
        return annotation.sample

def segment_signal(record:str,channel:list, segment_length_s:int):
        """
        Segment a WFDB record of a given channel into given lengths in seconds

        returns:
        stress_signal - a numpy array with the stress signal segments
        non_stress_signal - a numpy array with the non stress signal segments
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

        return non_stress_signal, stress_signal

def group_one_data(directory:str, channel:list, subject_number:int, segment_length:int):
        """
        Groups one subjects data into stress and non stress categories.
        Mostly a wrapper for the segment_signal function.

        returns:
        stress - a numpy array with segments(arrays) in the stress category
        non_stress - a numpy array with segments(arrays) in the non stress category
        """

        #figure out the name of the subject file and extract the data
        record_name = directory + "/Subject" + str(subject_number) +"_AccTempEDA"
        stress, non_stress = segment_signal(record=record_name, channel=channel, 
                                        segment_length_s=segment_length)
        return non_stress, stress

def group_all_data_by_segments(directory:str, channel:list, data_count:int, segment_length:int):
        """
        Groups all subject data into stress and non stress segments. 
        The data is grouped into these two categories, therefore it cannot be traced
        to a certain subject. 
        Propably mostly for intern use, as group_all_data_by_subject function has in 
        theory more functionality.

        Defines two global arrays stress_segments and non_stress_segments, through which 
        data can be accessed. 

        returns:
        None
        """

        #declare needed arrays
        global stress_segments
        global non_stress_segments
        stress_segments = []
        non_stress_segments = []

        #go through the subject data and group it into the stress and non stress categories
        for i in range(data_count):
                temp_stress, temp_non_stress = group_one_data(directory, channel, i+1, segment_length)
                non_stress_segments.extend(temp_non_stress)
                stress_segments.extend(temp_stress)

def group_all_data_by_subject(directory:str, channel:list, data_count:int, segment_length:int):
        """
        Groups all data by subject into stress and non stress segments

        Defines a global array subject_data, through which the subject data can be accessed 
        by subject_data[subject_number][stress], in which stress is 1 and non stress 0.

        returns:
        None
        """

        #declare needed arrays
        global subject_data
        subject_data = []
        temp_subject = []

        #go through the subject data and group it into stress and non stress by subject
        for i in range(data_count):
               temp_subject = group_one_data(directory, channel, i+1, segment_length)
               subject_data.append(temp_subject)

def get_subject_data(subject_number:int, stress:bool, segment_number=500):
        """
        A Function to facilitate getting subject data.

        If the function encounters an IndexError, the whole stress/non stress 
        category of the given subject. 
        The subject_number is expected to be given from 1,...,n, e.g. to 
        access the first subject you have to use 1 as the subject_number
        (not 0, as you would normally as a programmer)

        returns:
        specific segment or subject data
        """
        if (stress): i=1
        else: i=0
        try:
                return subject_data[subject_number - 1][i][segment_number]
        except IndexError:
                return subject_data[subject_number - 1][i]

def test_cases():
        #test for grouping data by segments
        group_all_data_by_segments(directory="data", channel=EDA, data_count=20, segment_length=30)
        a = len(stress_segments)
        b = len(non_stress_segments)
        print(a)
        print(b)
        print(a+b)

        for i in range(a):
                assert(len(stress_segments[i]) == 240)
        for i in range(b):
                assert(len(non_stress_segments[i]) == 240)
        print("finished grouping by segments")

        #test for grouping data by subject
        group_all_data_by_subject(directory="data", channel=EDA, data_count=20, segment_length=30)
        c = len(subject_data)
        print(c)
        assert(c==20)
        for i in range(20):
                for j in get_subject_data(i + 1,True):
                        assert(len(j) == 240)
                for k in get_subject_data(i + 1,False):
                        assert(len(k) == 240)
        print("finished grouping by subject")

test_cases()



