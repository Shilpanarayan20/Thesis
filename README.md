# Thesis
Thesis contains 4 Folders Dataset,(X,Y)_Co-ordinates,DataSet_1,Dataset_2
       
       Dataset :
       
	    This folder consist of 2 subfolders Dataset_1 and Dataset_2
	       1) Dataset_1 folder consist of 100 participants and Separate Folder for each participants that each separate folder contains:
	                * folder Small with Small Letters (a-z) (csv file along with the visualization of that character)
		        * folder Capital with Capital Letters (A-Z) (csv file along with the visualization of that character)
		        * folder Numbers with Numbers (0-9) (csv file along with the visualization of that character)
			
	       2) Dataset_2 folder consist of 55 participants and Separate Folder for each participants that each separate folder contains:
	                * Capital, Small Letters and Numbers (0-9, A-Z, a-z) (csv file along with the visualization of that character)
  		        * (Not_writing_sequence_start - writing_sequence - Not_writing_sequence_end)
			* Example data file - PersonID.Letter.sequence(Activity).csv
					       * PersonID  = P1.....P55
					       * Letter = 0-9, A-Z, a-z
					       * Sequence are W = Writing_Sequence, N1 = Not_Writing_Sequence_Start, N2 = Not_Writing_Sequence_End
					       * Activity includes * E = Eating
								   * M = Using Mobile
								   * S = Scratching Head, face, hand
								   * D = Drinking
 								   * OS = Placing object sidewise
								   * OD = Placing object upside down
								   * EG = Removing eye glasses
								   * Random daily movements
								   

       (X,Y)_Co-ordinates :
       
	        This folder consist of 2 subfolders Dataset_1 and Dataset_2
		    1) Dataset_1 folder consist of 100 participants and Separate Folder for each participants that each separate folder consist of (X,Y)Co-Ordinates calculated or extracted using raw sensor data in writing sequences 
		    
		    2) Dataset_2 folder consist of 55 participants and Separate Folder for each participants that each separate folder consist of (X,Y)Co-Ordinates calculated or extracted using raw sensor data of Not_writing_sequence_start - writing_sequence - Not_writing_sequence_end 
		    
		    
        DataSet_1 :
	 
	       This folder consist of 2 subfolders Train_Set and Test_set
	           1) Train_Set again contains Two sub folders 1a). (X,Y)_Co-ordinates and 1b). Sensor 
		             1a). (X,Y)_Co-ordinates : 
			                  Contains 70 participants and each participants data is stored in separated folder id and that folder contains the(X,Y)Co-Ordinates calculated or extracted using raw sensor data in writing sequences
					  
			     1b). Sensor : 
			                  Contains 70 participants and each participants data is stored in separated folder id and that folder contains the  slected raw feature data taken from raw sensor data in writing sequences
					  
		   2) Test_Set again contains Two sub folders 1a). (X,Y)_Co-ordinates and 1b). Sensor 
		             1a). (X,Y)_Co-ordinates : 
			                  Contains 30 participants and each participants data is stored in separated folder id and that folder contains the  (X,Y)Co-Ordinates calculated or extracted using raw sensor data in writing sequences
					  
			     1b). Sensor : 
			                  Contains 30 participants and each participants data is stored in separated folder id and that folder contains the  slected raw feature data taken from raw sensor data in writing sequences			  
					  
		
	   DataSet_2 :
	
	       This folder consist of 2 subfolders Train_Set and Test_set
	           1) Train_Set again contains Two sub folders 1a). (X,Y)_Co-ordinates and 1b). Sensor 
		             1a). (X,Y)_Co-ordinates : 
			                  Contains 38 participants and each participants data is stored in separated folder id and that folder contains the  (X,Y)Co-Ordinates calculated or extracted using raw sensor data in Not_writing_sequence_start - writing_sequence - Not_writing_sequence_end 
					  
			     1b). Sensor : 
			                  Contains 38 participants and each participants data is stored in separated folder id and that folder contains the  slected raw feature data taken from raw sensor data in Not_writing_sequence_start - writing_sequence - Not_writing_sequence_end 
					  
		   2) Test_Set again contains Two sub folders 1a). (X,Y)_Co-ordinates and 1b). Sensor 
		             1a). (X,Y)_Co-ordinates : 
			                  Contains 17 participants and each participants data is stored in separated folder id and that folder contains the  (X,Y)Co-Ordinates calculated or extracted using raw sensor data in Not_writing_sequence_start - writing_sequence - Not_writing_sequence_end 
					  
			     1b). Sensor : 
			                  Contains 17 participants and each participants data is stored in separated folder id and that folder contains the  slected raw feature data taken from raw sensor data in Not_writing_sequence_start - writing_sequence - Not_writing_sequence_end 		  
					  
					  
