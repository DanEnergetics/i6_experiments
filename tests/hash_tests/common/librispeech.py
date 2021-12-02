from i6_experiments.common.datasets.librispeech import export_all
export_all(output_prefix="datasets")

jobs = [
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.HUlwJhUVuhW3',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.JXZDXuAtXDFI',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.JvhHpRn0rBdm',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.NIf5NG6AJsbD',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.OXnLLhLdj9ew',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.OZrQakm9Lizy',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.R3T4Yu1U9wHG',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.TldQiAWb5RHL',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.UAWVU1qN3OjJ',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.VRZTadgWvOJb',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.cVZtp3w3ifs1',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.da4jvvvtTyUG',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.gjPZrjES7C9A',
    'work/i6_core/audio/encoding/BlissChangeEncodingJob.vUdgDkgc97ZK',
    'work/i6_core/corpus/stats/ExtractOovWordsFromCorpusJob.AQnble9lAUXO',
    'work/i6_core/corpus/stats/ExtractOovWordsFromCorpusJob.Hde5OgrOSgj5',
    'work/i6_core/corpus/stats/ExtractOovWordsFromCorpusJob.clziMIB5yqmZ',
    'work/i6_core/corpus/stats/ExtractOovWordsFromCorpusJob.e4HC80X6mhgT',
    'work/i6_core/corpus/stats/ExtractOovWordsFromCorpusJob.qTfzKn4SsFi5',
    'work/i6_core/corpus/stats/ExtractOovWordsFromCorpusJob.zwqpPoBp69uA',
    'work/i6_core/corpus/transform/MergeCorporaJob.3TLb8BOVqgHz',
    'work/i6_core/corpus/transform/MergeCorporaJob.MFQmNDQlxmAB',
    'work/i6_core/corpus/transform/MergeCorporaJob.NY7Vhl4yKxwG',
    'work/i6_core/corpus/transform/MergeCorporaJob.cYuETsPYTfaz',
    'work/i6_core/corpus/transform/MergeCorporaJob.hlZ8ixhLSaaQ',
    'work/i6_core/corpus/transform/MergeCorporaJob.jDrChrheFmOE',
    'work/i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.4LL17D9Sz7NZ',
    'work/i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.8iqSB1tz3OMD',
    'work/i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.CeoY6kKnh8B3',
    'work/i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.KQqvdreEkegi',
    'work/i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.QLzK7S51OV6m',
    'work/i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.gWxOHrcGHjNZ',
    'work/i6_core/datasets/librispeech/DownloadLibriSpeechCorpusJob.o9cbE9VC7lgn',
    'work/i6_core/datasets/librispeech/DownloadLibriSpeechMetadataJob.n7Yd9EbtVi13',
    'work/i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.2GMuOxuirZVL',
    'work/i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.HzYJupd3Ufg2',
    'work/i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.N4devEBOAvgK',
    'work/i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.OjEfOC2QXh8l',
    'work/i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.h78m0D0Rx6uQ',
    'work/i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.qlLkwjdH203i',
    'work/i6_core/datasets/librispeech/LibriSpeechCreateBlissCorpusJob.wjSkfzJS1Ge2',
    'work/i6_core/g2p/apply/ApplyG2PModelJob.0dGBUol5JvuO',
    'work/i6_core/g2p/apply/ApplyG2PModelJob.GJhGsRwbb8Nn',
    'work/i6_core/g2p/apply/ApplyG2PModelJob.IV0dFT7SuO2P',
    'work/i6_core/g2p/apply/ApplyG2PModelJob.b3RfOcGqJIBP',
    'work/i6_core/g2p/apply/ApplyG2PModelJob.k7kR4vWMXz02',
    'work/i6_core/g2p/apply/ApplyG2PModelJob.sMrFAmEzVi6Y',
    'work/i6_core/g2p/convert/BlissLexiconToG2PLexiconJob.HHFogGVyjvg1',
    'work/i6_core/g2p/convert/BlissLexiconToG2PLexiconJob.Say8hHODLnt6',
    'work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.0ipHRfHW2EOP',
    'work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.3kvfvZJVSVVO',
    'work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.JOqKFQpjp04H',
    'work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.YOCGTFAqEzXW',
    'work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.ooQBGEX46hrM',
    'work/i6_core/g2p/convert/G2POutputToBlissLexiconJob.uvYfrdFPhIqB',
    'work/i6_core/g2p/train/TrainG2PModelJob.GnXFuV8mtbXb',
    'work/i6_core/g2p/train/TrainG2PModelJob.VcBIBCQtkqMK',
    'work/i6_core/lexicon/conversion/LexiconFromTextFileJob.A3IBmxcidywQ',
    'work/i6_core/lexicon/conversion/LexiconFromTextFileJob.mTRl42KFeZSx',
    'work/i6_core/lexicon/modification/MergeLexiconJob.0E1tsFpfctIb',
    'work/i6_core/lexicon/modification/MergeLexiconJob.z54fVoMlr0md',
    'work/i6_core/lexicon/modification/WriteLexiconJob.ssLyWABKo3vf',
    'work/i6_core/returnn/oggzip/BlissToOggZipJob.5ad18raRAWhr',
    'work/i6_core/returnn/oggzip/BlissToOggZipJob.Cbboscd6En6A',
    'work/i6_core/returnn/oggzip/BlissToOggZipJob.FxDdqM5eyngA',
    'work/i6_core/returnn/oggzip/BlissToOggZipJob.NSdIHfk1iw2M',
    'work/i6_core/returnn/oggzip/BlissToOggZipJob.RvwLniNrgMit',
    'work/i6_core/returnn/oggzip/BlissToOggZipJob.VN8PpcLm5r4s',
    'work/i6_core/returnn/oggzip/BlissToOggZipJob.W2k1lPIRrws8',
    'work/i6_core/returnn/oggzip/BlissToOggZipJob.aEkXwA7HziQ1',
    'work/i6_core/returnn/oggzip/BlissToOggZipJob.uJ4Bsi72tTTX',
    'work/i6_core/text/processing/PipelineJob.p4BOP5qZ6T1G',
    'work/i6_core/tools/download/DownloadJob.0UXAqd5DuQG7',
    'work/i6_core/tools/download/DownloadJob.6ij8dDC1z4zK',
    'work/i6_core/tools/download/DownloadJob.ddCQ2YhC3VG0'
]

