#!/usr/bin/env python
# encoding=utf-8

"""
use accelerator only
 
fea:modOnly [10,1]
mod get smoothed by middle-value
confidence given by each obs prediction,all 30 trees vote majority percentage
stroll
"""

 

dataPath='/home/yr/magnetic_fea/data1014/' 

para_pack=0#1with liangbin riding into trainset 
import numpy as np
import pylab as plt
import cPickle,math,random,theano
import theano.tensor as T
import lasagne
from leancloud import Object
from leancloud import Query
import leancloud
import time,os,operator,scipy


deviceId={'huawei':'ffffffff-c7a8-3cd1-ffff-ffffea16e571', 
	'xiaomi':'ffffffff-c7a7-c7d4-0000-000031a92fbe', 
	'mine':'ffffffff-c95d-196f-0000-00007e282437',
	'vivo':'ffffffff-c43f-692d-ffff-fffff4d57110',
	'wangx':'ffffffff-c28a-1bf0-0000-00005e774605',
	'zhangx':'ffffffff-f259-8a54-0000-0000300c3e8c',
	'liyt':'ffffffff-c7a8-3c91-ffff-ffffa2dc21fc',
	'donggy':'ffffffff-c95e-eae5-ffff-ffffce758366',
	'hangwz':'ffffffff-c43f-77d4-0000-00001b78f081',
	'zishuai':'ffffffff-9475-8052-ffff-ffffaa303f0b',
	'test_meizu':'ffffffff-910d-998f-0000-0000017e0bce',
	'test1':'ffffffff-c7a8-3bf6-0000-0000062f2428',
	'biany':'ffffffff-c28b-7966-0000-000049d75b84',
	'liuxuan':'ffffffff-f259-8a2f-0000-0000040de9c1',
	'wangzhenfang':'00000000-53a7-c4ef-ffff-fffffd609c24',
	'cuiyajie':'ffffffff-910d-998f-0000-0000585e3d7e',
	'zishuai_1':'ffffffff-9475-8052-ffff-ffffaa303f0b'
	}


###################
c_list=['running','walking','riding','sitting','driving']
class_type=c_list[4]
##################
month_=12;day_=4
nov_vivo=[[5,16,6,5,16,8],[6,13,28,6,13,37],[6,15,41,6,15,45],[7,17,4,7,17,15]]
nov_test=[[11,6,16,31,11,6,16,36],[11,9,16,25,11,9,16,30],[11,9,16,37,11,9,17,27],[11,9,20,34,11,9,20,50],\
	[11,12,17,50,11,12,18,30],[11,13,17,0,11,13,17,20],[11,14,14,15,11,14,14,27],[11,14,15,19,11,14,15,50],[11,14,16,30,11,14,16,41],[11,14,16,58,11,14,17,25],[11,14,18,29,11,14,18,34],[11,16,16,23,11,16,18,0],[month_,day_,19,7,month_,day_,19,23]]


inst_id_zs='mTl3M9NJ0qjkKOKFX378VWa37IFYKBe5'
inst_id_hjw='rumE011he7vtJxkINHkHQTdhkoBjJMcr'
inst_id_vivo='czte5wJCwAFRSUJWsC5ybyaezAhDnOm1'
###########query label.data
device=deviceId['vivo'] 
period=nov_test[-1]

###########query log.tracer
inst_id=inst_id_zs
#period=watch_nov_zs[-1]
 
 
#############33

def generate_stamp(period):
	#[8,28,8,33]->[(2015, 10, 20, 22, 30, 0, 0, 0, 0),(2015, 10, 20, 22, 48, 0, 0, 0, 0)]->stamp
	dur= [(2015, period[0], period[1], period[2], period[3], 0, 0, 0, 0),\
		(2015, period[4], period[5], period[6], period[7], 0, 0, 0, 0)]
	stamp_range0=[time2stamp(dur[0]),time2stamp(dur[1])]
	stamp_range=[t*1000 for t in stamp_range0]
	return stamp_range 


def connect_db_log():##sensor.log.tracer   not label.data, 
	import leancloud,cPickle
	appid = "9ra69chz8rbbl77mlplnl4l2pxyaclm612khhytztl8b1f9o"
	   
	appkey = "1zohz2ihxp9dhqamhfpeaer8nh1ewqd9uephe9ztvkka544b"
	#appid = "ckjjqzf3jqwl8k1u4o10j2hqp0rm0q6ferqcfb0xpw00e8zl"
	#appkey = "rn0qw8ib96xl0km63kylo6v5afuxclc8jti5ol8rx4kylqob"
	leancloud.init(appid, appkey)

def time2stamp(t):
	#t = (2015, 9, 28, 12, 36, 38, 0, 0, 0)
	stamp = int(time.mktime( t )) ;
	return stamp
def connect_db_label():#label.data
	import leancloud,cPickle
	appid = "ckjjqzf3jqwl8k1u4o10j2hqp0rm0q6ferqcfb0xpw00e8zl"
	appkey = "rn0qw8ib96xl0km63kylo6v5afuxclc8jti5ol8rx4kylqob"
	leancloud.init(appid, appkey)

 
 
def get_content_fromLabel(results):#result is from find() 
 	 
	obs={}; 
	r=results
	for i in range(1):
		#print type(r.get("events")) 
		if len(r.get("events"))>=1:
			 
			print r.get("motion"),r.get("events").__len__()
			ll=r.get("events") #ll=[ {},{}...]
			for dic in ll[:]:
			#dic={timestamp:xxxx,value:[1,2,3]...}{u'timestamp':xxxx,value:[1,2,3]...}
				#print '1',dic
			
			#print dic["timestamp"],' ',dic["values"][0],' ',dic["values"][1],' ',dic["values"][2]
				 
				if 'timestamp' in dic and dic["timestamp"] not in obs.keys():
					obs[ dic["timestamp"] ]=[r.get("motion"),\
					dic["values"][0],dic["values"][1],dic["values"][2]  ]
				###data form: {timestamp:[obs],...}  [obs]=[motion,x,y,z]
		 
	 
	return obs 

 

#####################
def get_content_fromLog(results):#result is <>
 	 
	obs={}; 
	r=results
	
	ll=r.get("value") #ll=[ {},{}...]
	for dic in ll["events"][:]:#dic={timestamp:xxxx,value:[1,2,3]...}
			
		#print dic["timestamp"],' ',dic["values"][0],' ',dic["values"][1],' ',dic["values"][2]
		if dic["timestamp"] not in obs.keys():
			obs[ dic["timestamp"] ]=[class_type,\
			dic["values"][0],dic["values"][1],dic["values"][2]  ]
			###data form: {timestamp:[obs],...}  [obs]=[motion,x,y,z]
		 
	###########################
	print 'final',obs.__len__()
	###################3
	return obs 
	 


def get_all(query,skip,result):
	limit=500
	query.limit(limit)
	query.skip(skip)
	found=query.find()
	if found and len(found)>0:
		result.extend(found)
		print 'av_utils get all,now result len:',len(result),'skip',skip
		return get_all(query,skip+limit,result)
	else:
		return result
	


def save2pickle(c,name):
    write_file=open(dataPath+str(name),'wb')
    cPickle.dump(c,write_file,-1)#[ (timestamp,[motion,x,y,z]),...]
    write_file.close()
 
def load_pickle(path_i):
    f=open(path_i,'rb')
    data=cPickle.load(f)#[ [time,[xyz],y] ,[],[]...]
    f.close()
    #print data.__len__(),data[0]
    return data	







def fea4(obs):#[50,]obs
	#4
	mean=np.mean(obs);std=np.std(obs)
	min_i=np.min(obs);max_i=np.max(obs)
	f=np.array([mean,std,min_i,max_i])#[4,]
	dim=obs.shape[0]
	#percentile 5
	percentile=[10/100.*dim,25/100.*dim,50/100.*dim,75/100.*dim,90/100.*dim];#print percentile
	perc=[int(i) for i in percentile];#print perc
	obs_sort=np.sort(obs)#[50,]
	perc_i=obs_sort[perc];#print perc_i#[5,]
	gap4=perc_i[3]-perc_i[1]
	gap5=perc_i[4]-perc_i[0]
	#sum, square-sum 12
	position=[5,10,25,75,90,95]
	pos=[int(i/100.*dim) for i in position];#print pos
	sum_i=[np.sum(obs_sort[:i]) for i in pos]#[5,]
	sqrt_sum_i=[np.sqrt(np.dot(obs_sort[:i],obs_sort[:i])) for i in pos]
	#
	fea_i=np.concatenate((f,perc_i,gap4.reshape((1,)),gap5.reshape((1,)) ),axis=0);#print fea_i.shape
	return fea_i[:]#[10,]
###############################################


def classify(inputTree,testVec):#[17,] [13,]
	firstStr = inputTree.keys()[0]#[dim,value]
	dim1,v1=firstStr
	secondDict = inputTree[firstStr]
   
     
	if testVec[dim1] <=v1:#go left
        	if type(secondDict['left']).__name__ == 'dict':
			
                	classLabel = classify(secondDict['left'], testVec)
            	else: 
			classLabel = secondDict['left']
			 
	else:#go right
		if type(secondDict['right']).__name__ == 'dict':
                	classLabel = classify(secondDict['right'], testVec)
            	else: 
			classLabel = secondDict['right']
			 
				
	return classLabel


"""
def normalize(x):#[n,12]
	num,dim_x=x.shape
	min_each_dim=np.min(x,axis=0);print 'normaliz',min_each_dim.shape#[12,]
	max_each_dim=np.max(x,axis=0);
	x1=(x-min_each_dim)/(max_each_dim-min_each_dim);print 'norm',x1.shape
	return x1
"""



def predict_ensemble(X_test,stumps_model):#[1,5] |   [[i],[]...]  [i]=[tree,dim_list,accuracy]
	stumps=stumps_model
	n_val=X_test.shape[0];dim_val=X_test.shape[1]-1#5=fea4+1label
	f_label_mat=np.zeros((n_val,stumps.__len__())) #[n,10stump]  
	n_stumps=float(len(stumps))

	for ind in range(stumps.__len__()):#[3,5,8,11..]  [tree,dimList,accuracy]
		dim_sample=stumps[ind][1]
		tree=stumps[ind][0]
		for obs in range(n_val):
			pred=classify(tree,X_test[obs,:])#5=4+1
			f_label_mat[obs,ind]=pred


	##
	#f_label majority vote
	 
	maj_vote=np.zeros((n_val,))#[n,]
	for i in range(f_label_mat.shape[0]): #[n,10stump]
		vote1=f_label_mat[i,:].sum()#num of 1
		vote0=(1-f_label_mat[i,:]).sum()#num of 0
		maj_vote[i]=[1 if vote1>vote0 else 0][0]
		confidence=[vote1/n_stumps if vote1>vote0 else vote0/n_stumps][0]
	######
	return sum(maj_vote)/float(n_val),confidence #



def predict_eachObs(X_test3,X_test10,X_test3modxyz):#[1,5]  [1,5] [1,45]
	#######init label
	label=-1
	confidence=1
	label_watch=-1
	#############
	# stroll drivesit 1 | walkrunrid 0
	###################
	
	#####ensemble test
	prob_ds,ds_confidence=predict_ensemble(X_test10,stumps_ds_wrr) 
	print 'drivesit percentage',prob_ds
	if prob_ds>0.5:print 'not-final drive sit',prob_ds,ds_confidence
	elif prob_ds<0.5:print'not-final walkrunrid',1-prob_ds,ds_confidence
	#
	confidence=ds_confidence
	############
	#stroll 0| drivesit1
	################
	if prob_ds>0.5:
		
		#acc no_mag, no_normalize x,magnetic=noise,accelerator only
		#####ensemble test
		ds_prob,ds_confidence1=predict_ensemble(X_test3,stumps_s_ds)#[1,5]
		notwat_prob,notwat_confidence=predict_ensemble(X_test3modxyz,stumps_watchphone)
		print 'ds percentage',ds_prob
		if ds_prob>0.5:
			print 'final pred sit',ds_prob,ds_confidence1
			label='sitting'
			
		if ds_prob<0.5:
			print 'final pred stroll',1.-ds_prob,ds_confidence1
			label='stroll'
		#####watchphone
		if notwat_prob>0.5:label_watch='notwatch'
		if notwat_prob<=0.5:label_watch='watch'
		
		
		###
		confidence*=ds_confidence1


	############
	#runwalk 0|rid1
	################
	if prob_ds<0.5:
		#####ensemble test
		rid_prob,rid_confidence=predict_ensemble(X_test3,stumps_r_rw)#[1,13]
		notwat_prob,notwat_confidence=predict_ensemble(X_test3modxyz,stumps_watchphone)
		print 'rid percentage',rid_prob
		if rid_prob>=0.5:
			print 'final pred rid',rid_prob,rid_confidence
			label='riding'
			#####watchphone
			if notwat_prob>0.5:label_watch='notwatch'
			else:
				label_watch='watch';label='walking'
			
		elif rid_prob<0.5:
			print 'not-final pred runwalk',rid_confidence
		#
		confidence*=rid_confidence


		############
		#run 0|walk1
		################
		if rid_prob<0.5:
			#####ensemble test
			walk_prob,walk_confidence=predict_ensemble(X_test3,stumps_w_r)#[1,13]
			notwat_prob,notwat_confidence=predict_ensemble(X_test3modxyz,stumps_watchphone)
			print 'walk percentage',walk_prob
			if walk_prob>=0.5:
				print 'final pred walk',walk_prob,walk_confidence
				label='walking'
				#####watchphone
				if notwat_prob>0.5:label_watch='notwatch'
				else:label_watch='watch'
			elif walk_prob<0.5:
				print 'final pred run',1-walk_prob,walk_confidence
				label='running'
				#####watchphone
				if notwat_prob>0.5:label_watch='notwatch'
				else:
					label_watch='watch';label='walking'
				
			#
			confidence*=walk_confidence

	
	return label,label_watch,confidence


def fft(mod):#[50,]
	y=np.abs(scipy.fftpack.fft(mod) )#[50,]
	return y.reshape((-1,1))#[50,1]



def mid_smooth(mod,wind_sz):#[n,]
	#wind_sz=10
	n=mod.shape[0];print 'n',n
	 
	mod1=[]
	for i in range(n)[:-wind_sz]:
		patch=mod[i:i+wind_sz]#[3,]
		pi=np.sort(patch)[int(wind_sz/2)]
		mod1.append(pi)
	#
	patch=mod[-wind_sz:]
	pi=np.sort(patch)[1]
	for t in range(wind_sz):
		mod1.append(pi)
	##
	 
	print np.array(mod1).shape[0]
	return np.array(mod1)


def print_label_mat(label_list):
	c_dic={'running':0,'walking':1,'riding':2,'sitting':3,'driving':4,'stroll':5}
	rst_dic={}
	for num in label_list:
		if num not in rst_dic:
			rst_dic[num]=1
		else: rst_dic[num]+=1
	for label,num in c_dic.items():
		print label,rst_dic[num]
		
		

############3
if __name__=="__main__":
	"""
	###########
	#db log.tracer
	################
	####init
	connect_db_log()
	log = leancloud.Object.extend('Log')
	log_query = leancloud.Query(log)
	#print 'all',log_query.count()##error
	inst_query = leancloud.Query(leancloud.Installation)
	print 'install',inst_query.count()#2335
	inst = inst_query.equal_to('objectId', inst_id).find();#print '1',inst[0]
	#
	#################each period
	all_record_list=[];i=0
	####period
	stamp_range=generate_stamp(period)
	###########
	#query  acc
	#################
	log_query.equal_to('installation', inst[0]).equal_to("type",'accSensor').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count acc',log_query.count()
		 
	######get all
	acc_list=[]; all_acc_list=[]
	if log_query.count()>0:
		record_list=get_all(log_query,0,[]);#print 'all',len(record_list),record_list[0]
		#instance object[<>,<>,<>,...]
		######each record
		
		for obj in record_list[:]:
			obs=get_content_fromLog(obj)#data form: {timestamp:[o],...}  [o]=[motion,x,y,z]
			all_acc_list.append(obs)#[{},...]
	###############combine{} sort by timestamp
	data_dic={};acc_list=all_acc_list
    	for dic in acc_list:
		for k,v in dic.items():
			if k not in data_dic:
				data_dic[k]=v
	##sort acc
	ll=sorted(data_dic.items(),key=lambda f:f[0],reverse=False)
	# # DATA FORMATE  {timestamp:[motion x y z],...}->[ (timestamp,[motion,x,y,z]),...]
	xyz=np.array([obs[1][1:] for obs in ll]);print 'xyz',xyz.shape#[ [xyz],[]...] shape[n,3]
	###########
	#query  mag
	#################
	log_query.equal_to('installation', inst[0]).equal_to("type",'magneticSensor').\
	less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])
	print 'record count mag',log_query.count()
		 
	######get all
	mag_list=[]; all_mag_list=[]
	if log_query.count()>0:
		record_list=get_all(log_query,0,[]);#print 'all',len(record_list),record_list[0]
		#instance object[<>,<>,<>,...]
		######each record
		
		for obj in record_list[:]:
			obs=get_content_fromLog(obj)#data form: {timestamp:[o],...}  [o]=[motion,x,y,z]
			all_mag_list.append(obs)#[{},...]
	###############combine{} sort by timestamp
	data_dic={};mag_list=all_mag_list
    	for dic in mag_list:
		for k,v in dic.items():
			if k not in data_dic:
				data_dic[k]=v
	##sort acc
	ll=sorted(data_dic.items(),key=lambda f:f[0],reverse=False)
	# # DATA FORMATE  {timestamp:[motion x y z],...}->[ (timestamp,[motion,x,y,z]),...]
	xyz_mag=np.array([obs[1][1:] for obs in ll]);print 'xyzmag',xyz_mag.shape#[ [xyz],[]...] shape[n,3]
	###save
	save2pickle([xyz,xyz_mag],'accmag-xyz-'+class_type)	 
	""" 



  	  
	#########################3
	#db label.data
	#########################3
	####init
	connect_db_label()
	stamp_range=generate_stamp(period) 
	########
	#query acc
	##########
	UserSensor = Object.extend('UserSensor')
	query_acc = Query(UserSensor) 
	
	query_acc.equal_to("deviceId",device).not_equal_to("events",None).\
		equal_to("sensorType",'accelerometer').\
		less_than("timestamp", stamp_range[1]).greater_than("timestamp",stamp_range[0])

	print 'count',query_acc.count() 
	acc_list=get_all(query_acc,0,[]);print 'acc',len(acc_list)#,all_list[0]#instance object
	
	#########3
	all_acc_list=[];i=0
	for obj in acc_list:
		obs=get_content_fromLabel(obj)#data form: {timestamp:[o],...}  [o]=[motion,x,y,z]
		all_acc_list.append(obs)#[{},...]
		i+=obs.__len__()
	print 'total',i,all_acc_list.__len__()
	 
	#####
	###############combine{} sort by timestamp
	data_dic={};acc_list=all_acc_list
    	for dic in acc_list:
		for k,v in dic.items():
			if k not in data_dic:
				data_dic[k]=v
	##sort acc
	ll=sorted(data_dic.items(),key=lambda f:f[0],reverse=False)
	# # DATA FORMATE  {timestamp:[motion x y z],...}->[ (timestamp,[motion,x,y,z]),...]
	xyz=np.array([obs[1][1:] for obs in ll]);print 'xyz',xyz.shape#[ [xyz],[]...] shape[n,3]
	##clean
	#xyz=np.concatenate((xyz[:522,:],xyz[827:,:]),axis=0)#[n,3]
	#xyz=xyz[300:,:]
	save2pickle(xyz,'test-acc-xyz-'+class_type);print 'save',xyz.shape
	  
	
	

	 
	##########################################
	#  visual,test
	#################################################
	#xyz_acc=load_pickle(dataPath+'test-acc-xyz-'+class_type+'-WRONGDRIVE-jiaweislow')#[n,3]  -1  -WRONGWALK
	#xyz_acc=load_pickle(dataPath+'test-acc-xyz-'+class_type+'-1113')#[n,3]
	xyz_acc=load_pickle(dataPath+'test-acc-xyz-'+class_type)


	###test all patch in one go
	#xyz_acc=load_pickle('/home/yr/magnetic_fea/data1027/testvivo-acc-xyz-sitting')#stroll sitting
	#########################
	#some error report
	##############
	mod_acc_mean=np.mean( np.sqrt( (xyz_acc*xyz_acc).sum(axis=1) ) )
	assert isinstance(xyz_acc,np.ndarray) 
	assert xyz_acc.shape[0]>=50 
	assert xyz_acc.shape[1]==3
	assert mod_acc_mean>=5 and mod_acc_mean<=50
	 
	##########################3
	##clip data clean
	#xyz_acc=xyz_acc[150:,:]
	#save2pickle(xyz_acc,'normal-acc-xyz-'+class_type)	 
	#########
	abs_xyz_acc=np.abs(xyz_acc)
    	mod_acc=np.sqrt( (xyz_acc*xyz_acc).sum(axis=1) );print 'mod',mod_acc.shape #[n,]
	mod_acc3=mid_smooth(mod_acc,3)
	mod_acc10=mid_smooth(mod_acc,10)
	#
	xyz_mod=np.concatenate((xyz_acc,mod_acc.reshape((-1,1))  ),axis=1)#[n,3],[n,1] ->[n,4]
	xyz_mod_smooth3=np.array([mid_smooth(xyz_mod[:,i],3) for i in range(xyz_mod.shape[1])])#[4,n]
	xyz_mod_smooth3=xyz_mod_smooth3.T#[n,4]
    	maxabs_acc=abs_xyz_acc.max(axis=1);#print 'maxvec',maxabs.shape#[n,]
    	minabs_acc=abs_xyz_acc.min(axis=1);#print 'minvec',minabs.shape#[n,]
	 
	#########visual
	plt.figure()
	plt.plot(xyz_acc[:,0],'r-',xyz_acc[:,1],'b-',xyz_acc[:,2],'g-')
	 
	ind1=range(mod_acc.shape[0]) 

    	plt.figure()
	plt.subplot(311);plt.title('acc-'+'mod3 mod10 mod')
    	plt.plot(ind1,mod_acc3,'ro',
		ind1,mod_acc10,'y-',
		ind1,mod_acc,'b--');
	plt.ylim(0,30);plt.xlim(0,3500); 

	 
	#plt.show()

	 
	###########################
	#generate obs x y [n,3mod]->[n_obs,50,3] ->[n_obs,4x3]
	############################
	 
	 
	kernel_sz=50.;stride=kernel_sz;
    	
    	num=int( (mod_acc3.shape[0]-kernel_sz)/stride ) +1
	  #[n,]
	####only acc mod fea4
	obs_list3=[]; obs_list10=[];obs_list3_xyz=[]
    	for i in range(num)[:]: #[0,...100] total 101 
        	obs3=mod_acc3[i*stride:i*stride+kernel_sz]#[50,]
		obs10=mod_acc10[i*stride:i*stride+kernel_sz]#[50,]
		obs3_xyzmod=xyz_mod_smooth3[i*stride:i*stride+kernel_sz,:]#[50,4]
		if obs3.shape[0]==kernel_sz:
			#smooth3
			v3=fea4(obs3)#[50,]->[4,]
			obs_list3.append(v3)#[50,]->[4,]
			#smooth10
			v10=fea4(obs10)
			obs_list10.append(v10)
			#mod xyz smooth3
			v=np.array([fea4(obs3_xyzmod[:,i]) for i in range(obs3_xyzmod.shape[1])]).flatten()#[50,4]->[44,]
			obs_list3_xyz.append(v)
	x_arr3=np.array(obs_list3)#[n,11]
	x_arr10=np.array(obs_list10);print 'xy',obs_list3.__len__()
	x_arr_modxyz3=np.array(obs_list3_xyz)#[n,44]
	
	
	 
	 
	

	 
	###load all model####
	stumps_ds_wrr=load_pickle('/home/yr/magnetic_fea/data1102-drivesit-walkrunrid/rf-para-drivesit-walkrunrid-modOnly')
	stumps_s_ds=load_pickle('/home/yr/magnetic_fea/data1101_drivesit/rf-para-stroll-drivesit-modOnly')
	stumps_r_rw=load_pickle('/home/yr/magnetic_fea/data1027/rf-para-rid-walkrun-modOnly')
	stumps_w_r=load_pickle('/home/yr/magnetic_fea/data1024_walkrun/rf-para-walkrun-modOnly')
	stumps_watchphone=load_pickle('/home/yr/watchphone/data1019/rf-para-watchphone-modOnly-smooth3')
	####each observation
	
	 
	dataSet3_mod=np.concatenate((x_arr3,np.zeros((x_arr3.shape[0],1)) ),axis=1)#[n,4] [n,1]->[n,5]
	dataSet10_mod=np.concatenate((x_arr10,np.zeros((x_arr10.shape[0],1)) ),axis=1)#[n,4] [n,1]->[n,5]
	dataSet3_modxyz=np.concatenate((x_arr_modxyz3,np.zeros((x_arr_modxyz3.shape[0],1)) ),axis=1)#[n,44] [n,1]->[n,5]
	
	#c_dic={'running':0,'walking':1,'riding':2,'sitting':3,'driving':4}
	c_dic={'running':0,'walking':1,'riding':2,'sitting':3,'driving':4,'stroll':5}
	c1_dic={'watch':0,'notwatch':1};#c_list2=['watch','notwat']in stumps
	#X_test=dataSet#[n,13]
	label_list=[];confidence_list=[];label_watch_list=[]
	for i in range(dataSet3_mod.shape[0]):
		 
		X_test3=dataSet3_mod[i,:].reshape((1,-1))#[5,]->[1,5]
		X_test10=dataSet10_mod[i,:].reshape((1,-1))#[5,]->[1,5]
		X_test3modxyz=dataSet3_modxyz[i,:].reshape((1,-1))#[45,]->[1,45]
		label_motion,label_watch,confidence=predict_eachObs(X_test3,X_test10,X_test3modxyz)#  ->string
		label_list.append(c_dic[label_motion])
		label_watch_list.append(c1_dic[label_watch])
		confidence_list.append(confidence)
	
	#############visual each obs_i 

	
	plt.subplot(312)
	plt.title('running=0,walking=1,riding=2,sitting=3,driving=4,stroll=5')
	plt.plot(label_list,'ro',label='running=0,walking=1,riding=2,sitting=3,driving=4,stroll=5')
	plt.xlabel('time');plt.ylabel('predict class');
	plt.subplot(313)
	plt.title('watch phone 0-watch  1-not')
	plt.plot(label_watch_list,'go')
	#plt.legend()
	plt.figure()
	plt.plot(confidence_list,'bo')
	plt.show()


 
	
	 


	
	
	 
	
	
	

	
	 
	 

	 

 

	 
	
	 

	
    
 
	
		
	
   		 



