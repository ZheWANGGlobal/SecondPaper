function judge=knear(k,sampleset,samplesign,testset)  %k��k�����㷨��kֵ��sampleset��������samplesign�����ı�ǩ��testset���Լ�
G=max(samplesign);
for i=1:size(testset,1);              %�Լ����������ÿ�������������²���
    for j=1:size(sampleset,1);           %�������������������������ŷʽ���루δ������
        res(j,1)=(sampleset(j,:)-testset(i,:))*(sampleset(j,:)-testset(i,:))';
    end
    num(1:G,1)=0;
    for n=1:k;            %Ѱ��k��res����С����ֵ��������������1��2��3�ĸ����ж�����������ڵڼ���
        a=res(1,1);
        b=1;
        for j=1:size(sampleset,1);
            if a>res(j,1)         %�趨aΪ��һ����ֵ��ѭ�������������ֵ����С��a=С����ֵ
                a=res(j,1);
                b=j;              %��¼���С��ֵ���ڵ�����
            end
        end
        res(b,1)=256;             %�ҳ������ֵ��λ���Ժ�ֵ256�������ظ�����
        num(samplesign(b,1),1)=num(samplesign(b,1),1)+1;
    end
    a=num(1,1);
    b=1;
    for j=1:G;
        if a<num(j,1)
           a=num(j,1);
           b=j;
        end
    end
    judge(i,1)=b;
          
end
%judge          %�жϽ���������