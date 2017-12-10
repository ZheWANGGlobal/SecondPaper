#include "mex.h"
#include <stdlib.h>
#include <math.h>
//������ʽ[judge,r,gailv]=cknear(k,sampleset,samplesign,testset)
//[judge,r]=cknear(k,sampleset,samplesign,testset)
//judge=cknear(k,sampleset,samplesign,testset)
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
    double *sampleset,*testset,*samplesign;
    double *K,*Ki,*gailv,*res,*num,*judge;   //judge��Ϊ��ż�������ı�־ .num�����������ڸ���ĵ�ĸ���
    double g,a;
	double *temp1,*temp2,*temp3;
    int M,N,G,len,k;       //gΪ����
    int i,j,r,b;
    if(nlhs>3||nrhs!=4)
    {
        mexErrMsgTxt("the number of input must be 4 and the number of output must be 3 or less than 3");
    }
    M=mxGetM(prhs[1]);  //ȡ����������Ĵ�С
    N=mxGetN(prhs[1]);
    len=mxGetM(prhs[3]);//len��ֵtestset����Ŀ
    sampleset=mxGetPr(prhs[1]);   //���������ݽ���ȡֵ
    samplesign=mxGetPr(prhs[2]);
    testset=mxGetPr(prhs[3]);
    g=0;
    for(i=0;i<M;i++)           //gȡ������־�����ֵ
    {
        if(samplesign[i]>g)
            g=samplesign[i];
    }
    G=(int)g;
    K=mxGetPr(prhs[0]);
    k=(int)K[0];                 //��ȡ��һ���������ݣ���ֵk
    Ki=(double *)mxMalloc(len*sizeof(double)); //���¶���Ki����Ϊ������뾶
    judge=(double *)mxMalloc(len*sizeof(double)); //�����жϾ���ռ�
    res=(double *)mxMalloc(M*sizeof(double)); //res���ŷʽ����
    gailv=(double *)mxMalloc(len*G*sizeof(double)); //gailv=mxCreateDoubleMatrix(len,G,mxREAL);//������ʾ���
    num=(double *)mxMalloc(G*sizeof(double)); //num[G]
    for(i=0;i<len;i++)                   //��ÿ��������ݽ��в���
    { 
        for(j=0;j<M;j++)
        {
           res[j]=0;
           for(r=0;r<N;r++)
              res[j]=res[j]+(sampleset[j+r*M]-testset[i+r*len])*(sampleset[j+r*M]-testset[i+r*len]);
        }
        for(j=0;j<G;j++)
           num[j]=0;
        //Ѱ��k���������룬����ĸ���num[j]
        for(j=0;j<k;j++)
        {
            a=res[0];
            b=0;
			while(a<=1.e-8)
			{				
				b=b++;
				a=res[b];				
			}
            for(r=0;r<M;r++)
            {
              if((a>res[r]) && (res[r]>1.e-8))
              {
                  a=res[r];
                  b=r;
              }
            }
            if(j==k-1)        //��k����С����Ϊ�뾶��ƽ����ֵKi
            {
                Ki[i]=sqrt(res[b]);
            } 
            res[b]=-1;    //�����Լ�ͳ��
            num[(int)samplesign[b]-1]=num[(int)samplesign[b]-1]+1;       
        }
        //����ÿ����ʣ���ֵ���ʾ���
        g=num[0];
        r=0;
        for(j=0;j<G;j++)
        {        
            gailv[i+j*len]=num[j]/k;
            if(g<num[j])       //Ѱ������num����r=j����j+1Ϊ��ʶ
            {
                g=num[j];
                r=j;
            }
        }
        judge[i]=r+1;    //��ֵ�жϾ���
    }

    if(nlhs==3)
    {
      plhs[0]=mxCreateDoubleMatrix(len,1,mxREAL);//���ٿռ�������󣬴���жϱ�־��m*1ά
      plhs[1]=mxCreateDoubleMatrix(len,1,mxREAL);//��ŵ�k�������ľ���
      plhs[2]=mxCreateDoubleMatrix(len,G,mxREAL);//����������ڸ���ĸ���
	  temp1 = mxGetPr(plhs[0]);
	  temp2 = mxGetPr(plhs[1]);
	  temp3= mxGetPr(plhs[2]);
	  for(i=0;i<len;i++)
	  { 
		   temp1[i] = judge[i];
		   temp2[i]=  Ki[i];
		  for(j=0;j<G;j++)
			  temp3[i+j*len]=gailv[i+j*len];		  
	  }
      //mxSetPr(plhs[0],judge);           //�����ֵ
     // mxSetPr(plhs[1],Ki);
      //mxSetPr(plhs[2],gailv);
    }
    else if(nlhs==2)
    {
       plhs[0]=mxCreateDoubleMatrix(len,1,mxREAL);//���ٿռ�������󣬴���жϱ�־��m*1ά
       plhs[1]=mxCreateDoubleMatrix(len,1,mxREAL);//��ŵ�k�������ľ���
	   temp1 = mxGetPr(plhs[0]);
	  temp2 = mxGetPr(plhs[1]);	
	  for(i=0;i<len;i++)
	  { 
		   temp1[i] = judge[i];
		   temp2[i]=  Ki[i];		  
		  
	  }
    //   mxSetPr(plhs[0],judge);           //�����ֵ
      // mxSetPr(plhs[1],Ki);
    }
    else
    {
        plhs[0]=mxCreateDoubleMatrix(len,1,mxREAL);//���ٿռ�������󣬴���жϱ�־��m*1ά
		temp1 = mxGetPr(plhs[0]);
	  for(i=0;i<len;i++)
	  { 
		   temp1[i] = judge[i];		   
	  }
        //mxSetPr(plhs[0],judge); 
    }
    // k=(int)K[0];                 //��ȡ��һ���������ݣ���ֵk
    //Ki=(double *)mxMalloc(len * sizeof(double));//���¶���Ki����Ϊ������뾶
    //judge=(double *)mxMalloc(len * sizeof(double));//�����жϾ���ռ�
    //res=(double *)mxMalloc(M * sizeof(double));//res���ŷʽ����
    //gailv=(double *)mxMalloc(len*G * sizeof(double));//gailv=mxCreateDoubleMatrix(len,G,mxREAL);//������ʾ���
    //num=(double *)mxMalloc(G * sizeof(double)); //num[G]
   mxFree(Ki);
   mxFree(judge);
   mxFree(res);
   mxFree(gailv);
   mxFree(num);
	return;
}