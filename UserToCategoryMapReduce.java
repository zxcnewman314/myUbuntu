import java.io.IOException;
import java.util.StringTokenizer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class UserToCategoryMapReduce {

    public static class UserToCategoryMapper extends Mapper<Object, Text, Text, Text> {

        private Text category = new Text();
        private Text user = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
            StringTokenizer tokenizer = new StringTokenizer(value.toString(), "\t");
            String userId = tokenizer.nextToken();
            while (tokenizer.hasMoreTokens()) {
                category.set(tokenizer.nextToken());
                user.set(userId);
                context.write(category, user);
            }
        }
    }

    public static class UserToCategoryReducer extends Reducer<Text, Text, Text, Text> {

        private Text users = new Text();

        public void reduce(Text key, Iterable<Text> values, Context context)
                throws IOException, InterruptedException {
            StringBuilder sb = new StringBuilder();
            for (Text value : values) {
                sb.append(value.toString()).append(" ");
            }
            users.set(sb.toString().trim());
            context.write(key, users);
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "user-to-category");
        job.setJarByClass(UserToCategoryMapReduce.class);
        job.setMapperClass(UserToCategoryMapper.class);
        job.setReducerClass(UserToCategoryReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
