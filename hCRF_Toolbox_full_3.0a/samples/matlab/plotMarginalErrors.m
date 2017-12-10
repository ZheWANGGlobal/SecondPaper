function plot_ll_errors(base_crf_rc,rt_crf_rcs)
%function plot_ll_errors(base_crf_rc,rt_crf_rcs)
%
%Plots the error between the given base_crf predictions and one or more
%real-time crfs.
%
%From this point I'll assume there is a rt_crf_rcs variable containing all the
%tests executed with real-time CRFs, and base_crf_rc variable containing the
%base example. All of them should also have saved log-likelihood during
%testing, since this is what we'll be comparing against.
%
%We will evaluate the average disparity, difference in marginal
%probabilities between the real-time CRF and the base CRF. Both were
%trained in a similar way.

%assert(strcmp(base_crf_rc.params.errorType, rt_crf_rcs{1}.params.errorType), 'Error types are not the same for both models, cant plot comparison of errors.');

num_rt_crf = size(rt_crf_rcs,2);
num_seqs = size(base_crf_rc.ll,2);

averageMarginalErrorTest = zeros(num_rt_crf,1);
delays = zeros(num_rt_crf,1);
errorsTest = zeros(num_rt_crf,1);
%Window size should be the same for everyone, otherwise this is
%meaningless.
for i=1:num_rt_crf
    delays(i) = rt_crf_rcs{i}.params.delay;
    errorsTest(i) = rt_crf_rcs{i}.testError;
    for seq_i=1:num_seqs
        averageMarginalErrorTest(i) = averageMarginalErrorTest(i) + mean(abs(base_crf_rc.ll{seq_i}(1,:) - rt_crf_rcs{i}.ll{seq_i}(1,:)));
    end
    averageMarginalErrorTest(i) = averageMarginalErrorTest(i) / num_seqs;
end

figure(1)
hold on
plot(delays,averageMarginalErrorTest);%,delays,averageMarginalErrorVal);
xlabel('Delay');
ylabel('Marginal prob. error');
title('Average error on marginal probabilities with varying delays');
legend('Test');%,'Validation');
hold off

figure(2)
hold on
plot(delays,errorsTest,'r');
%plot(delays,errorsVal,'b');
plot(0,base_crf_rc.testError,'ro','MarkerSize',5,'MarkerFaceColor','w');
%plot(0,base_crf_rc.validateError,'bo','MarkerSize',5,'MarkerFaceColor','w');
xlabel('Delay');
ylabel('Error');
title('Error with varying delays');
legend('Test','Base CRF Test');
hold off
