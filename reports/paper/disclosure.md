# Generated AI-use disclosure

This compliance record is generated deterministically from durable execution logs. It is not a recollection supplied by an author or model.

- Schema: `research_swarm.generated_disclosure.v1`
- Generation policy: `execution_logs_only`
- Event journal: `absent_not_fabricated`

## Models and tools by execution role

| Role | Tool/backend | Model | Version | Runs |
|---|---|---|---|---:|
| Operator | manual | gpt-5.4 | unlogged | 1 |
| Worker | codex | unlogged | unlogged | 18 |
| Worker | manual | gpt-5.4 | unlogged | 6 |
| Worker | manual | unlogged | unlogged | 2 |
| Worker | operator_backfill | unlogged | unlogged | 5 |

## Logged gate outcomes

- `make gate`: return code `0` observed 26 time(s).
- `make gate`: return code `2` observed 1 time(s).
- `make test`: return code `0` observed 1 time(s).
- `python scripts/release_assembly.py --release-date 2026-04-11 --check`: return code `0` observed 1 time(s).
- `python src/analysis/build_str_release_outputs.py --as-of 2026-04-09`: return code `0` observed 1 time(s).
- `python src/analysis/build_str_release_outputs.py --sample`: return code `0` observed 1 time(s).
- `python src/validation/validate_str_pipeline.py --as-of 2026-04-09`: return code `0` observed 1 time(s).

## Review and human-edit history

- Reviewer role `Judge` recorded outcome `approve` 21 time(s).
- Reviewer role `Judge` recorded outcome `block` 2 time(s).
- Provenance classification `backfill` appears in 5 annotation(s).
- Provenance classification `executor_run` appears in 18 annotation(s).
- Provenance classification `manual_operator` appears in 9 annotation(s).

## Disclosure statement

AI systems executed planning, implementation, analysis, review, and/or operational roles only as recorded above. The systems are disclosed assistants and are not authors. Every model/version value shown is taken from a durable log; `unlogged` means the version was not recorded. Scientific and release claims remain subject to repository gates and human-author consent requirements.

## Evidence inventory

- `reports/status/reviews/T000_legacy_backfill.json` — sha256 `feaac192b01359c9d0476d9358e21b5c363bc3c14713f5ace58a99bca56dcbfb`
- `reports/status/reviews/T005_legacy_backfill.json` — sha256 `de98642e8aa723ccfe78f0b6ad746fa6be59f2855b55f610d6e1f1fd5ba29885`
- `reports/status/reviews/T010_legacy_backfill.json` — sha256 `2077352e63bd83b95a50e81336a1a945038ecddefaa4609dc81ba1f91b20d026`
- `reports/status/reviews/T015_legacy_backfill.json` — sha256 `57d4e16ad9d432681aa4e7ac10818daa4d80cf816cf842172dd8ea3af28496f5`
- `reports/status/reviews/T020_legacy_backfill.json` — sha256 `37410ea6082cab7e54e344c826b9ebab01cd1d76703e73cbc8c1a5aa44c3c468`
- `reports/status/reviews/T025_20260401T113303Z.json` — sha256 `01c8ff7a532cd1177c22e51a72cdbe7bc06a42395d0fdd96d1914ee6b6258172`
- `reports/status/reviews/T030_20260401T121619Z.json` — sha256 `8b0e49132163b3fc812824d675cc103c264147051afc7cacdd89c8a880965341`
- `reports/status/reviews/T035_20260408T143620Z.json` — sha256 `4b947e6ed61aaa7bf0c19b692e6de0db4a8be35d0007b8c560b2d317ea82485f`
- `reports/status/reviews/T035_20260408T144724Z.json` — sha256 `84984976f7c96e2b984cf58494ac1b3d6a629f7a6f179da2774c42667cf1c318`
- `reports/status/reviews/T040_20260408T151504Z.json` — sha256 `386a6f46fdd39f1208966744f3b4f0018c43303fe9a90f703813cd0777ae2550`
- `reports/status/reviews/T045_20260408T164956Z.json` — sha256 `55a34917ae34beddd47246ee7fa24d09ec13ef4e259ecf0270f5f25593a5c00a`
- `reports/status/reviews/T046_20260409T112715Z.json` — sha256 `46998dc0818079fa8cd926ef06c331e760e4ba76b9a1b24cb4a1029fc0285e14`
- `reports/status/reviews/T047_20260408T191606Z.json` — sha256 `b1ffa1ecb15dc60df58e40d793333fd12bf3d831dee15bbb0732f229852720ab`
- `reports/status/reviews/T048_20260410T192135Z.json` — sha256 `bd016db3a31c1536df36a1d684b4d92e3513c7528d9d0481596c8adf13580246`
- `reports/status/reviews/T049_20260410T191826Z.json` — sha256 `28a1ce273026319fb6f2cf4917e835cc774f41ebd9c6e32f9429ca392e070055`
- `reports/status/reviews/T050_20260410T211201Z.json` — sha256 `6f61917d83877565ebd8e5140a84f3c85ca5beed9be6bf097420f09b40411f9a`
- `reports/status/reviews/T051_20260410T192822Z.json` — sha256 `7c0ebafaf74a2a0905719cbe22dda89b013841fcf3971050fd25e77091a5fbb5`
- `reports/status/reviews/T052_20260410T194240Z.json` — sha256 `b8c70accf83222fb2e4baae148e16e3a0ccddf3baaee5764086078af55be44f2`
- `reports/status/reviews/T060_20260411T141012Z.json` — sha256 `639986bee8ffb5ccb2b993a2fc260be8a8ccb56ebecb0a12a978b012eb442927`
- `reports/status/reviews/T070_20260411T153559Z.json` — sha256 `cf414eca99c2bc3a5fb7ccf01592849f8dae0f1039ceab49ef9172b0a5197abf`
- `reports/status/reviews/T080_20260411T161415Z.json` — sha256 `4c455db73cf9402523e197eff5b4c7776ed8238d35ded57b26be64345531aa93`
- `reports/status/reviews/T080_20260411T161524Z.json` — sha256 `93e434a68659212b0b471d7a34b08b87db815cd7a804e34c51100bd32517810e`
- `reports/status/reviews/T080_20260411T161630Z.json` — sha256 `108fa27608f66f830a6be5637f9a4666c0fe3be6872b9042f3aba142a91274cd`
- `reports/status/swarm_runs/T000_legacy_backfill.json` — sha256 `a6e142f7428e4aeac05d0453a3bdf26df34efc5cdb5c14eb0ca034ad09895502`
- `reports/status/swarm_runs/T005_legacy_backfill.json` — sha256 `25b5e032bf03b17723d330af35382faf8c37b319aa0536ece89de0bf2f600876`
- `reports/status/swarm_runs/T010_legacy_backfill.json` — sha256 `499aebc2aa1607f411ebc0837d61e458d74bcb6bb9f503553cc525a4ff4b577b`
- `reports/status/swarm_runs/T015_legacy_backfill.json` — sha256 `fbc26252dc5d7254691d8e52469e5069be52d4103704f2f6ff3197f28b39ea1f`
- `reports/status/swarm_runs/T020_legacy_backfill.json` — sha256 `dddf0177f925a4b3223cc50709442ed381fc63838985780e1af488b30c94e20a`
- `reports/status/swarm_runs/T025_20260331T230838Z.json` — sha256 `9375817bb6ae2449d04ce23f5b3d3b379c75a685de84b8f70bad2fc579e2ceea`
- `reports/status/swarm_runs/T030_20260401T115602Z.json` — sha256 `828b37f4cb5989da600b30add41de17e37fd9afb2efdd8375d4bcc6a36e9b7c1`
- `reports/status/swarm_runs/T035_20260401T135106Z.json` — sha256 `9bf9e1257a4a9ad2d810a8cd2d003ea35d9435b8e50169e0abf9f6d126152dc4`
- `reports/status/swarm_runs/T035_20260401T152122Z.json` — sha256 `3f7091017e06f9d4eca28f9242f82c411a2538fb55272a3636673e919eb34ef3`
- `reports/status/swarm_runs/T035_20260401T155307Z.json` — sha256 `46f27fa7e345c1fc2823a43c0dc2c57be577569438c708c97c53cbb49ea82497`
- `reports/status/swarm_runs/T035_20260401T163917Z.json` — sha256 `ed8fbe18e590eb14ea3bdfdea5f4070e8325ef5e371abcb24e3dec77e5422438`
- `reports/status/swarm_runs/T035_20260402T112730Z.json` — sha256 `2cdfc53e0d6d759bfffaeec8eefcfd70ff9b03d8b3412f409a945acb17e81d0f`
- `reports/status/swarm_runs/T035_20260403T145543Z.json` — sha256 `5b42de0decafd05127830b7c03a5af14b997ce0d4a7bf872f2aba77d00b8fa2e`
- `reports/status/swarm_runs/T035_20260408T142235Z.json` — sha256 `2f5172031cea3ca277a129356b33890af0b1e0ba96eec32a8512a12950083d01`
- `reports/status/swarm_runs/T035_20260408T144706Z.json` — sha256 `130e0d6d6c0f8d700a1d9121e45961233ad88ad56c00ce55de8206c4e3e7dabe`
- `reports/status/swarm_runs/T040_20260408T150755Z.json` — sha256 `8c4d31c13a10c084cfbbc9d3b6b5468e5428cd38e99669ba0f1be350baff748b`
- `reports/status/swarm_runs/T045_20260408T161648Z.json` — sha256 `bc9b7f049ae22f7f977eda569d689aa23ff081df947d6a74ba01bccc12cbc607`
- `reports/status/swarm_runs/T045_20260408T164746Z.json` — sha256 `04fa5cd8d7cf2969ee610805fcd1eaabb50d1973312e9a867b71ce2a9c2f8983`
- `reports/status/swarm_runs/T045_20260408T164811Z.json` — sha256 `e96b9b269f2de0b0d916c8e3f59374c33ada5d150455370287472762ee75afbf`
- `reports/status/swarm_runs/T046_20260408T165337Z.json` — sha256 `58caa46f5f8edab45aff15cf9aef4bf0096aaaf4e417dbc59b8c6c94e8e6effc`
- `reports/status/swarm_runs/T046_20260408T211156Z.json` — sha256 `733f8b658956fb1b3f71cb76aae4c3cdcd31507f197463c365b794e1353a34d4`
- `reports/status/swarm_runs/T047_20260408T170541Z.json` — sha256 `f7bfe378f00f54bfcde7038d0877aae8b03b3a856af430ba598cabeebb606451`
- `reports/status/swarm_runs/T048_20260410T192109Z.json` — sha256 `3a1ca1cd6c71894eaa0a5ded5a3e2de93c66d07523811441027fc6e4526dfb12`
- `reports/status/swarm_runs/T049_20260410T191750Z.json` — sha256 `1f10ca3df83a7e404f3ec9401d023d473a2b1bc803ee2747f54195814e91929a`
- `reports/status/swarm_runs/T050_20260410T211117Z.json` — sha256 `215cb8f209f714effa971b588b4d13d6bb86fbbd8445424052b6f0f5e234adaa`
- `reports/status/swarm_runs/T051_20260410T192759Z.json` — sha256 `ffc393ea46572aa20cfab9dcc411f88b5c02491cbe339a2bcfb227727abf2ec5`
- `reports/status/swarm_runs/T052_20260410T194203Z.json` — sha256 `08097a50db1cd0cd9fa998c427a43f25678fb7835ffe366a8c64348629f50814`
- `reports/status/swarm_runs/T060_20260411T113419Z.json` — sha256 `956de099b5d29ca1a51c8cf15cdf9c2be2e411100b69eb876953494ba49ad368`
- `reports/status/swarm_runs/T060_20260411T140930Z.json` — sha256 `eb16e87f7bcfdd24f5900e7eacbf838000f1a4bdb5516c933023abc5619a7fab`
- `reports/status/swarm_runs/T070_20260411T150935Z.json` — sha256 `58b2f4e69f57ca2c4b9198d59175b8d281d916782974e8d85550bc5fb371dd6b`
- `reports/status/swarm_runs/T070_20260411T153246Z.json` — sha256 `2a722d6d1216389df20751dd926e70f8f3b7d5eca146ede4a9d2a0ef440b6ccd`
- `reports/status/swarm_runs/T080_20260411T161258Z.json` — sha256 `81954fe528e7f402a092b264abb6249c04dcf55426038cd3025412daa2dbcbc1`
- `reports/status/swarm_runs/annotations/T000_legacy_backfill.json.provenance.json` — sha256 `992bfb37c5e1f70f1241252432d313449b0d528a8754d99728f8a5aebdfc9740`
- `reports/status/swarm_runs/annotations/T005_legacy_backfill.json.provenance.json` — sha256 `d64189649c50dd576d0518b861a3374ad24b6db610bd085ff8d05db8e15bbccc`
- `reports/status/swarm_runs/annotations/T010_legacy_backfill.json.provenance.json` — sha256 `a0eef3a2e121ea73a73cd3dcd18872513d9629b4c073ee868a0f9490bf67beea`
- `reports/status/swarm_runs/annotations/T015_legacy_backfill.json.provenance.json` — sha256 `e176f8623ac27321108425c5086ab6060ed4a460f56267e9bf6f7b196c5f3837`
- `reports/status/swarm_runs/annotations/T020_legacy_backfill.json.provenance.json` — sha256 `d67fb592107fc3c18727c0ce0c37c651c0b73c2d247ad2590a927328270ff2bd`
- `reports/status/swarm_runs/annotations/T025_20260331T230838Z.json.provenance.json` — sha256 `6b3fed439991a0c6318dcddf56345405d1b97b11c98c0d849f78bcdd833649a1`
- `reports/status/swarm_runs/annotations/T030_20260401T115602Z.json.provenance.json` — sha256 `3e9ae2e2a558895d9db0acaaa52576eac135b0a3275792ecb1b2d8b2fbdf205e`
- `reports/status/swarm_runs/annotations/T035_20260401T135106Z.json.provenance.json` — sha256 `e6bdcd3e2b3d70623e793735571cc02f5e35082990cdfa1d0205b3affb9a7d4d`
- `reports/status/swarm_runs/annotations/T035_20260401T152122Z.json.provenance.json` — sha256 `12cc5b42687bac633e5ba0434723145369bc4b2f021b35cd020beb4df58495c4`
- `reports/status/swarm_runs/annotations/T035_20260401T155307Z.json.provenance.json` — sha256 `45ce4f7f6f6ab6d896988a99ece0cb48fdd9f0411cea945f33159c3cd3193067`
- `reports/status/swarm_runs/annotations/T035_20260401T163917Z.json.provenance.json` — sha256 `ff78661037f4ede13380212d2f6c96d69203e972bb97aec1d99928db5286780e`
- `reports/status/swarm_runs/annotations/T035_20260402T112730Z.json.provenance.json` — sha256 `87fd9a6db66d82533d650bcdf25cb894f26cc2cdd6abbdf440734f2dfc203aa2`
- `reports/status/swarm_runs/annotations/T035_20260403T145543Z.json.provenance.json` — sha256 `415620f2a9d74a705d7aab4fdb9efb8d2693e8ae8db34bae6bfa9e701c0ffb11`
- `reports/status/swarm_runs/annotations/T035_20260408T142235Z.json.provenance.json` — sha256 `26a53538cd39a793ff372cc96f2ef85af5b3656d09e26126a22f2afb1dd1adea`
- `reports/status/swarm_runs/annotations/T035_20260408T144706Z.json.provenance.json` — sha256 `1152fd2c24620a1fc69d7abacd8efc32f9bec2fb03eec1345175cf61a38c15e9`
- `reports/status/swarm_runs/annotations/T040_20260408T150755Z.json.provenance.json` — sha256 `7857f7f50a833ef5a3d6d9a2fbcd008215cfcada07d30ca3bbe41a1fd04639df`
- `reports/status/swarm_runs/annotations/T045_20260408T161648Z.json.provenance.json` — sha256 `f494f339daae5db02a2c438141d11981e24a291b83daea196c39fc5113ebfc06`
- `reports/status/swarm_runs/annotations/T045_20260408T164746Z.json.provenance.json` — sha256 `fbf14bc178136535e9853971e5724914d9f89a3e017f9050c117296534c30dd6`
- `reports/status/swarm_runs/annotations/T045_20260408T164811Z.json.provenance.json` — sha256 `dceabb445de394f1aaed2ec2426cf5f8e6950b43581f219365b01bd9309e2c97`
- `reports/status/swarm_runs/annotations/T046_20260408T165337Z.json.provenance.json` — sha256 `b7d955e77c486f4cec960521b552dda8889d29c5a9981f40b2b2d0c5a8dd1e9c`
- `reports/status/swarm_runs/annotations/T046_20260408T211156Z.json.provenance.json` — sha256 `94c62a5ff9f1166be8b1d099d54538ca5888578a8027161c5739e1acca48dc9c`
- `reports/status/swarm_runs/annotations/T047_20260408T170541Z.json.provenance.json` — sha256 `2dbf9fabecfec1c270cec88349548aa50cd86479abc0da3df9d38e1350c521d7`
- `reports/status/swarm_runs/annotations/T048_20260410T192109Z.json.provenance.json` — sha256 `d662e9c987d15ce3399bd4df8cfb2d5a962a0833037f6712978c75d6b54214d5`
- `reports/status/swarm_runs/annotations/T049_20260410T191750Z.json.provenance.json` — sha256 `5b6c19d8eb8f0515b954c3ae3a44da7e364428c201b11cf9ef2aed18de3fd92a`
- `reports/status/swarm_runs/annotations/T050_20260410T211117Z.json.provenance.json` — sha256 `adc7b01c044c022b783dac1e58b4420942a996311077e06cc03f0c1458b9bfbd`
- `reports/status/swarm_runs/annotations/T051_20260410T192759Z.json.provenance.json` — sha256 `ccd0e30415cd77c8f532797fc06c01d0f6ea9a84c51450ad7d2de6f3aec91842`
- `reports/status/swarm_runs/annotations/T052_20260410T194203Z.json.provenance.json` — sha256 `bb85ead7576700af0c6c51dd5b984a19cc43bb73a88e14111e92af4dbc416391`
- `reports/status/swarm_runs/annotations/T060_20260411T113419Z.json.provenance.json` — sha256 `87106a290110218bca4330594eefd20647efff5d5427b9ab71315e0621a1ac10`
- `reports/status/swarm_runs/annotations/T060_20260411T140930Z.json.provenance.json` — sha256 `0e5b02707b3a8ab96b12795b90dd2c106401216494abc1d15cbb429901d9840b`
- `reports/status/swarm_runs/annotations/T070_20260411T150935Z.json.provenance.json` — sha256 `6efb6d5b4feed890f5de265121c945c897a3b42877bac04075f95768711e3110`
- `reports/status/swarm_runs/annotations/T070_20260411T153246Z.json.provenance.json` — sha256 `5f6a76f803b51da760308cf7b4b5f8fbaee4f143c86f8654e731c64ff51353fa`
- `reports/status/swarm_runs/annotations/T080_20260411T161258Z.json.provenance.json` — sha256 `2aff31df138e2854808e2217bf988bd1fa40c9e3479f837c718cfe088ac90941`
