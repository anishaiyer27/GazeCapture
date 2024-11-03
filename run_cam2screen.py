import matlab.engine
eng = matlab.engine.start_matlab()
eng.cam2screen(nargout=0)
eng.quit()
