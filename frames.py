## python script for getting clip from whole video

import ffmpeg
import sys
import datetime

sys.path.append(r'C:\\ffmpeg-2024-01-04-git-33698ef891-full_build\\bin')
with open('Data/t20.txt','r') as f:
    data = f.readlines()

for ball in data:
	info = ball.split(' ')
	start_time = info[1] # Start time for trimming (HH:MM:SS)
	end_time = info[2] # End time for trimming (HH:MM:SS)
	s_times = start_time.split(":")
	e_times = end_time.split(":")
	if int(s_times[2])>=2:
		new_sec = int(s_times[2])-2
		if new_sec <10:
			start_time = s_times[0]+":"+s_times[1]+":0"+str(new_sec)
		else:
			start_time = s_times[0] + ":" + s_times[1] + ":" + str(new_sec)
	if int(e_times[2])<58:
		new_end_sec = int(e_times[2]) + 2
		if new_end_sec < 10:
			end_time = e_times[0] + ":" + e_times[1] + ":0" + str(new_end_sec)
		else:
			end_time = e_times[0] + ":" + e_times[1] + ":" + str(new_end_sec)
	print(info[0],start_time,end_time)
	stream = ffmpeg.input("Data/ind_pak_t20.mp4")
	pts = "PTS-STARTPTS"
	stream = stream.trim(start = start_time,end = end_time).setpts(pts)
	output = ffmpeg.output(stream,"Data/t20/"+info[0]+'.mp4')
	output.run()

