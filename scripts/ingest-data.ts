import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import glob from 'glob';
import { OpenAIEmbeddings } from 'langchain/embeddings';
import { PineconeStore } from 'langchain/vectorstores';
import { pinecone } from '@/utils/pinecone-client';
import { TextLoader } from 'langchain/document_loaders';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';

(async () => {
  try {
    const files: string[] = await new Promise((resolve, reject) =>
      glob("docs/**/*.txt", (err, files) => err ? reject(err) : resolve(files))
    );

    console.log(`Found ${files.length} text files to ingest`);

    const embeddings = new OpenAIEmbeddings();
    const index = pinecone.Index(PINECONE_INDEX_NAME);

    for (const [i, file] of files.entries()) {
      console.log(`Ingesting file ${i + 1} of ${files.length}: ${file}`);
      try {
        const loader = new TextLoader(file);
        const rawDocs = await loader.load();
        const textSplitter = new RecursiveCharacterTextSplitter({
          chunkSize: 1000,
          chunkOverlap: 200,
        });
        const docs = await textSplitter.splitDocuments(rawDocs);
        await PineconeStore.fromDocuments(
          index,
          docs,
          embeddings,
          'text',
          PINECONE_NAME_SPACE,
        );
        console.log(`Ingested file ${i + 1} of ${files.length}: ${file}`);
      } catch (error) {
        console.error(`Failed to ingest file ${i + 1} of ${files.length}: ${file}: ${error}`);
      }
    }

    console.log('Ingestion complete');
  } catch (error) {
    console.error(`Failed to ingest your data: ${error}`);
  }
})();