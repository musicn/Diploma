import matlab.engine

eng = matlab.engine.start_matlab()

eng.cd(r'D:\FRI\Diploma\src\MDEC-master') # change directory to the location of the demo_1.m file

result = eng.demo_1(nargout=0) # call the function and return nothing

eng.quit()