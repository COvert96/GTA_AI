## GTA_AI ##

WORK IN PROGRESS -- still need to clean up this repository a lot -- many thanks to Sentdex for the amazing work.

Self-driving AI for GTA V

Most of the code has been taken from https://github.com/Sentdex/pygta5, I have just been adjusting the way the data is collected and processed.

1. Collect data ---> simply type in your command terminal (in the correct directory) python collect_data.py
    - The program will read whether you have a waypoint on or not and decide which data folder to append.
    - This will continue to be the folder until the process is paused (with 'T' key) and once unpaused it will check again.
    - The difference to Sentdex's is that mine collects any number of keys being pressed and collects all these options as categories before being encoded (giving roughly 9 more categories if you like to handbrake turn etc.)
    
2. Balance data
    - Remove all the extra straight keys as these skew the input data.
    - Remove the really rare categories, if you accidentally press three or four keys at once for example this is unnecessary to input into the model.
    
3. Train model
    - Still working out the kinks, my GPU struggles to process everything
    
4. Test model
    - Have not worked on this portion yet
